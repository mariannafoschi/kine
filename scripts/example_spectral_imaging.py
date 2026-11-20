# Copyright (C) 2026 Antonio Fuentes

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Spectral imaging with `kine`.

This file is a comprehensive example on how `kine` can be used to
reconstruct the Stokes I emission of a source at several frequencies, from
simultaneous VLBI observations in different frequency bands.

Two emission models are available, selected by ``spectral_model`` in the
parameter file:

- ``free`` (default): a neural field taking the frequency as a third input
  coordinate (f, x, y), with no explicit spectral model.
- ``powerlaw`` / ``curved``: a neural field taking the space coordinates (x, y) 
  and predicting per-pixel spectral parameters (I0, alpha[, beta]); the emission 
  at each frequency follows a power law (with optional curvature). This also 
  yields the spectral-index map alpha as a direct output.

References:
    [1] Foschi, M., Zhao, B., Fuentes, A. et al., "Video reconstruction of
    variable interferometric observations with neural fields." Under rev. (2026)
"""

import os
import glob
import queue
import argparse
import warnings
from collections import OrderedDict as odict
from concurrent.futures import ThreadPoolExecutor

import yaml
import numpy as np
import ehtim as eh
import ehtim.const_def as ehc
from tqdm import tqdm

import jax
import optax
from flax import linen as nn
from jax import numpy as jnp

import kine.utils as ut
import kine.field as fi
import kine.model as mo
import kine.obsdata as ob
import kine.trainer as tr

# Filter warnings
warnings.filterwarnings('ignore')

# Initialize async worker
q = queue.Queue()
ut.init_worker(fi.Video.async_plot, q)

# ______________________________________________________________________________
# Load hyperparameters and set RNG keys

# Load arguments from command line
par = argparse.ArgumentParser()
par.add_argument('-obs', type=str, help='uvfits file name')
par.add_argument('-yml', type=str, help='hyperparameter file name')
par.add_argument('-out', type=str, required=True, help='output folder name')
par = par.parse_args()

# Load hyperparameters from YAML file
with open(par.yml, 'r') as f:
    h = yaml.safe_load(f)
h = ut.HyperParams(h)

# Set RNG keys
rkey = jax.random.PRNGKey(h.seed)
np.random.seed(h.seed)

# Make output directory if it does not exist
outdir = os.path.join(par.out, '')
os.makedirs(outdir, exist_ok=True)

# Emission model
spectral = h.spectral_model in ('powerlaw', 'curved')

# ______________________________________________________________________________
# Load and pre-process data

obslist, bad = [], []
obspath = sorted(glob.glob(par.obs + '*.uvfits'))

print("-------\nLoading multi-frequency data...\n")

for path in tqdm(obspath):
    try:
        with ut.no_print():
            obs = ob.Obsdata.load_uvfits(path)
            obs = obs.avg_coherent(h.tavg)
            obs = obs.add_fractional_noise(h.syserr)
            obslist.append(obs)
    except Exception as e:
        print(f'Could not load {path}: {e} \n')
        bad.append(path)

# Fix metadata and empty array scans (hacky fix)
for i, _ in enumerate(obslist):
    obslist[i].fix_multifreq(obslist[-1])

# Get light curve (or zbl flux density)
lcurve = jnp.array([ob.get_zbl() for ob in obslist])

# ______________________________________________________________________________
# Set up grid of input coordinates and data products

# Field of view
fov = h.fov_uas * eh.RADPERUAS

# Set frequency coordinates
freqs = jnp.array([obs.rf for obs in obslist])
labels = ['B1 (214 GHz)', 'B2 (216 GHz)', 'B3 (226 GHz)', 'B4 (228 GHz)']
nfreqs = len(freqs)
print("Frequencies: ", freqs/1e9, 'GHz')

# Set coordinate grids: 3D (f, x, y) for the free field, 2D (x, y) for the
# spectral model, where the frequency enters through lognu
if spectral:
    grid = ut.get_grid(h.npix, h.npix)
    lognu = ut.get_lognu(freqs, h.nu_ref_ghz * 1e9)
else:
    grid = ut.get_grid(h.npix, h.npix, nfreqs, times=freqs)

# Set polarization channels (total intensity only)
outdim = 1

# Set empty image for image metadata and dimensions
improxy = eh.image.make_square(obslist[0], h.npix, fov, pol_prim='I')

# Compute lists of data products (target), uncertainties (sigma),
# and Fourier matrix (A) at each observed time
print("-------\nRetrieving data products from each epoch...\n")

data = {}
for dtype in h.data_prod:
    target, sigma, padmask = ob.Obsdata.get_data_nfft(obslist, dtype, improxy)
    data[dtype] = {
        'target': target,
        'sigma': sigma,
        'padmask': padmask
    }
    del target, sigma, padmask

# ______________________________________________________________________________
# Prepare stuff for NUFFT computations

print("-------\nSetting up NUFFT variables...\n")

def prepare_nufft(ob):
    """Helper function for parallelization."""
    bl = ob.get_baselines_nfft()
    uv = ob.get_uvpoints(improxy.psize)
    uvind = ob.get_uvpoints(improxy.psize, conj=False)
    pulse = ob.get_pulsefac(uv, ehc.PULSE_DEFAULT)
    tria = ob.get_closure_indices(bl, which='triangles')
    quad = ob.get_closure_indices(bl, which='quadrangles')
    return uv['u'], uv['v'], uvind['u'], pulse, tria, quad

# Threading pool
with ThreadPoolExecutor() as ex:
    u, v, uvind, pulse, tria, quad = map(
        list, zip(*ex.map(prepare_nufft, obslist))
    )

# Pad results to create JAX arrays
uv = {'u': ut.pad(u).astype('float32'),
      'v': ut.pad(v).astype('float32')}
uvind = ut.pad(uvind).astype('float32')
uvind = ut.map_val_to_ind(uv['u'], uvind)
pulse = ut.pad(pulse).astype('complex64')
tria = ut.pad(tria).astype('int32')
quad = ut.pad(quad).astype('int32')

# ______________________________________________________________________________
# Set up the network and training scheme

# ..................
# Initialize network

if spectral:
    # Spectral model: positional encoding of (x, y), i.e. no frequency
    network = mo.NeuralFieldSpectral(
        posenc_deg=tuple(h.nposenc[1:]),
        outdim=3 if h.spectral_model == 'curved' else 2,
        depth=h.depth,
        width=h.width,
        activ=nn.gelu,
        outactiv=nn.softplus,
        outshift=h.outshift,
        scaling_i=h.scaling_i,
        alpha_range=h.alpha_range,
        remat=h.remat
    )
else:
    # Free field: positional encoding of (f, x, y)
    network = mo.NeuralField(
        posenc_deg=tuple(h.nposenc),
        outdim=outdim,
        depth=h.depth,
        width=h.width,
        activ=nn.gelu,
        outactiv=nn.softplus,
        outshift=h.outshift,
        scaling_i=h.scaling_i,
        remat=h.remat
    )
variables = network.init(rkey, jnp.ones([grid.shape[-1]]), train=True)

# Schedule and train state
sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter: 0.1}
)
state = tr.Trainer.create(
    apply_fn=network.apply,
    params=variables['params'],
    batch_stats=variables['batch_stats'],
    tx=optax.adamax(sched)
)

# ______________________________________________________________________________
# Initialize the neural field

print("-------\nInitializing network...\n")

# .....................
# Create initialization

# Free field: one tophat per band. Spectral model: one tophat for I0 at the
# reference frequency and a flat spectrum.
if spectral:
    init_freqs = jnp.array([h.nu_ref_ghz * 1e9])
    init_labels = [f'I0 ({h.nu_ref_ghz} GHz)']
    init_lcurve = jnp.array([lcurve.mean()])
else:
    init_freqs, init_labels, init_lcurve = freqs, labels, lcurve

init_vid = fi.Video(
    init_freqs,
    h.npix,
    fov,
    obs.ra,
    obs.dec,
    h.initniter,
    dates=init_labels
)
init_vid.add_tophat(init_lcurve, h.init_params)
init_vid.plot(outpath=outdir+'init_image.png')

# .......................
# Initialization training

# Initialize video and loss
init = fi.Video(
    init_freqs,
    h.npix,
    fov,
    obs.ra,
    obs.dec,
    h.initniter,
    dates=init_labels
)
lloss, loss = [], 0

# Looping over epochs
for i in (pbar := tqdm(range(1, h.initniter+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, _, out, state = tr.Trainer.train_step(
        odict(
            state=state,
            grid=grid,
            init_arr=init_vid.iarr
        )
    )
   
    # Save loss
    lloss.append(loss)

    # Save video
    if i == 1 or i % 500 == 0 or i == h.initniter:
        q.put(
            dict(
                video=init,
                out=out[..., :1],
                loss=lloss,
                outpath=outdir+'output_init.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Training

print("-------\nTraining...\n")

# Update NPIX for training with NUFFT
tr.NPIX = h.npix

# ..............
# Traininig loop

# Initialize video and loss
video = fi.Video(freqs, h.npix, fov, obs.ra, obs.dec, h.niter, dates=labels)
lloss, loss = {dp: [] for dp in h.data_prod} | {'lcurve': []}, 0

# Training inputs
inputs = odict(
    grid=grid,
    data=data,
    lcurve=lcurve,
    uvpoints=uv,
    pulsefac=pulse,
    uvind=uvind,
    triangles=tria,
    quadrangles=quad
)
if spectral:
    inputs['lognu'] = lognu

# Looping over epochs
for i in (pbar := tqdm(range(1, h.niter+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, ldict, out, state = tr.Trainer.train_step(
        odict(state=state) | inputs
    )

    # Save loss
    for l in lloss:
        lloss[l].append(ldict[l])

    # Save video
    if i == 1 or i % 500 == 0 or i == h.niter:
        q.put(
            dict(
                video=video,
                out=out,
                loss=lloss,
                outpath=outdir+'output_train.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Save video and model parameters

# Visualize
print("-------\nSaving results...\n")
video.from_video(out, loss=lloss)
video.plot_gif(outpath=outdir+'output_multifreq.gif')
video.save_h5(outdir+'output_multifreq.h5')

# Generate output video on a finer grid
print("-------\nSampling network on a finer grid and saving results...\n")
video = fi.Video(freqs, h.npix_out, fov, obs.ra, obs.dec, h.niter, dates=labels)
if spectral:
    # Spectral model: frames and spectral-index map from the 2D (x, y) grid
    grid = ut.get_grid(h.npix_out, h.npix_out)
    spec = fi.SpectralVideo(
        ut.get_times_multiepoch(obslist[:1]), freqs, h.npix_out, fov,
        obs.ra, obs.dec, h.niter,
        dates=ut.get_times_multiepoch(obslist[:1], labels=True)
    )
    spec.from_state(state, grid, lognu=lognu, npix_train=h.npix)
    spec.plot_spectral_index(outpath=outdir+'output_alpha.png')
    video.iarr = spec.iarr
else:
    grid = ut.get_grid(h.npix_out, h.npix_out, nfreqs, times=freqs)
    video.from_state(state, grid, npix_train=h.npix)
video.save_h5(outdir+'output_multifreq2.h5')
video.plot_gif(outpath=outdir+'output_multifreq2.gif')
video.plot(outpath=outdir+'output_train2.png')

