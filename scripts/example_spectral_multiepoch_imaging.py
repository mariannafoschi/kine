# Copyright (C) 2026 Marianna Foschi, Antonio Fuentes, Brandon Zhao

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

"""Multi-frequency multi-epoch imaging with `kine`.

This file is a comprehensive example on how `kine` can reconstruct a sequence of 
Stokes I frames at different epochs and different frequencies, from a set of 
VLBI observations where each epoch is observed at several simultaneous frequency 
bands. Missing (epoch, band) combinations are allowed: those frames are 
interpolated by the network.

Two emission models are available, selected by ``spectral_model`` in the
parameter file:

- ``free`` (default): a free 4D neural field taking the frequency as a
  fourth input coordinate (t, f, x, y). All frames are reconstructed with
  a shared network but no explicit spectral coupling.
- ``powerlaw`` / ``curved``: a spectral neural field taking the space-time
  coordinates (t, x, y) and predicting per-pixel spectral parameters
  (I0, alpha[, beta]); the emission at each frequency follows a power law
  (with optional curvature). This also yields the spectral-index maps
  alpha as a direct output.

Total intensity only.

References:
    [1] Foschi, M. et al., In preparation. (2027)
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
from astropy.time import Time
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
ut.init_worker(fi.SpectralVideo.async_plot, q)

# ______________________________________________________________________________
# Load hyperparameters and set RNG keys

# Load arguments from command line
par = argparse.ArgumentParser()
par.add_argument('-obs', type=str, help='uvfits file name/prefix')
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

# Free fitting or spectral modeling
spectral = h.spectral_model in ('powerlaw', 'curved')

# ______________________________________________________________________________
# Load and pre-process data

obslist, bad = [], []
obspath = sorted(glob.glob(par.obs + '*.*'))

print("""\n-------\nLoading multi-frequency multi-epoch data...\n""")

for path in tqdm(obspath):
    with ut.no_print():
        obs = ob.Obsdata.load_uvfits(path)
        obs = obs.avg_coherent(h.tavg)
        obs = obs.add_fractional_noise(h.syserr)
        obslist.append(obs)

print(f"""\nTotal Files: {len(obspath)}\n""")

# Make metadata match exactly across epochs
for i, _ in enumerate(obslist):
    obslist[i].fix_multifreq(obslist[0])
    if obslist[i].scans is not None and len(obslist[i].scans) == 0:
        obslist[i].scans = None

# ______________________________________________________________________________
# Group observations onto the (epoch, frequency) frame grid

# Frame k is epoch k // nf and band k % nf
epochs, freqs, bands, table, mask = ut.group_multiepoch_multifreq(
    obslist, ut.get_times_multiepoch(obslist), freq_tol=h.freq_tol
)
dates = [Time(float(e), format='mjd').iso[:10] for e in epochs]
nt, nf = len(epochs), len(freqs)
nframes = nt * nf

# Fill missing frames with a placeholder observation
refob = next(o for o in table if o is not None)
obstab = [o if o is not None else refob for o in table]

# Get zbl flux density from each frame
lcurve = jnp.array([o.get_zbl() for o in obstab])

# ______________________________________________________________________________
# Set up grid of input coordinates and data products

# Field of view
fov = h.fov_uas * eh.RADPERUAS

# Set coordinate grids: (t, x, y) spectral model, or (t, f, x, y) free field
if spectral:
    grid = ut.get_grid(h.npix, h.npix, nt, times=epochs, tdil=h.tdil)
    lognu = ut.get_lognu(freqs, h.nu_ref_ghz * 1e9)
else:
    grid = ut.get_grid(
        h.npix, h.npix, nt, times=epochs, tdil=h.tdil,
        nf=nf, freqs=freqs, fdil=h.fdil
    )

# Set empty image for image metadata and dimensions
improxy = eh.image.make_square(refob, h.npix, fov, pol_prim='I')

# Compute lists of data products (target), uncertainties (sigma),
# and padding masks at each frame
print("""\n-------\nRetrieving data products from each frame...\n""")

data = {}
for dtype in h.data_prod:
    target, sigma, padmask = ob.Obsdata.get_data_nfft(obstab, dtype, improxy)
    data[dtype] = {
        'target': target,
        'sigma': sigma,
        'padmask': padmask * mask[:, None]
    }
    del target, sigma, padmask

# ______________________________________________________________________________
# Prepare stuff for NUFFT computations

print("""\n-------\nSetting up NUFFT variables...\n""")

def prepare_nufft(ob):
    """Get NUFFT variables for one observation."""
    bl = ob.get_baselines_nfft()
    uv = ob.get_uvpoints(improxy.psize)
    uvind = ob.get_uvpoints(improxy.psize, conj=False)
    pulse = ob.get_pulsefac(uv, ehc.PULSE_DEFAULT)
    tria = ob.get_closure_indices(bl, which='triangles')
    quad = ob.get_closure_indices(bl, which='quadrangles')
    return uv['u'], uv['v'], uvind['u'], pulse, tria, quad

# Get NUFFT variables in parallel
with ThreadPoolExecutor() as ex:
    u, v, uvind, pulse, tria, quad = map(
        list, zip(*ex.map(prepare_nufft, obstab))
    )

# Pad results to create equal shape JAX arrays
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

if spectral: # Spectral model
    network = mo.NeuralFieldSpectral(
        posenc_deg=(h.nposenc[0],) + tuple(h.nposenc[2:]),
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
else: # Free 4D field
    network = mo.NeuralField(
        posenc_deg=tuple(h.nposenc),
        outdim=1,
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

# .....................
# Create initialization

if spectral: # init_vid has 1 frame per epoch with frequency-averaged flux
    init_lcurve = (lcurve * mask).reshape(nt, nf).sum(axis=1) \
        / mask.reshape(nt, nf).sum(axis=1)
    init_vid = fi.Video(epochs, h.npix, fov, refob.ra, refob.dec, h.initniter)
else: # init_vid has 1 frame per (epoch, band)
    init_lcurve = lcurve
    init_vid = fi.Video(
        jnp.arange(nframes), h.npix, fov, refob.ra, refob.dec, h.initniter
    )
init_vid.add_tophat(init_lcurve, h.init_params)

# .......................
# Initialization training

# Initialize video and loss
if spectral:
    init = fi.Video(
        epochs, h.npix, fov, refob.ra, refob.dec, h.initniter, dates=dates
    )
else:
    init = fi.SpectralVideo(
        epochs, freqs, h.npix, fov, refob.ra, refob.dec, h.initniter, dates=dates
    )
lloss, loss = [], 0

# Looping over epochs
print("""\n-------\nInitializing network...\n""")

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
    if i == 1 or i % 1000 == 0 or i == h.initniter:
        q.put(
            dict(
                video=init,
                out=out[..., :1],
                loss=lloss,
                scale='log',
                drange=1e3,
                outpath=outdir+'init.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Training

# Update NPIX for training with NUFFT
tr.NPIX = h.npix

# ..............
# Traininig loop

# Initialize video and loss
video = fi.SpectralVideo(
    epochs, freqs, h.npix, fov, refob.ra, refob.dec, h.niter, dates=dates
)
lloss, loss = {dp: [] for dp in h.data_prod} | {'lcurve': []}, 0

# Training inputs; passing lognu selects the spectral model in the trainer
inputs = odict(
    grid=grid,
    data=data,
    lcurve=lcurve,
    mask=mask,
    uvpoints=uv,
    pulsefac=pulse,
    uvind=uvind,
    triangles=tria,
    quadrangles=quad
)
if spectral:
    inputs['lognu'] = lognu

# Looping over epochs
print("""\n-------\nTraining...\n""")

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
    if i == 1 or i % 1000 == 0 or i == h.niter:
        q.put(
            dict(
                video=video,
                out=out,
                loss=lloss,
                scale='log',
                drange=1e3,
                outpath=outdir+'out.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Save video, spectral index, and model parameters

print("""\n-------\nSampling network on a finer grid and saving results...\n""")

# Re-set grid up for output video
if spectral:
    grid = ut.get_grid(h.npix_out, h.npix_out, nt, times=epochs, tdil=h.tdil)
else:
    grid = ut.get_grid(
        h.npix_out, h.npix_out, nt, times=epochs, tdil=h.tdil,
        nf=nf, freqs=freqs, fdil=h.fdil
    )

# Generate output video
video = fi.SpectralVideo(
    epochs, freqs, h.npix_out, fov, refob.ra, refob.dec, h.niter, dates=dates
)
video.from_state(
    state, grid, lognu=lognu if spectral else None, loss=lloss,
    npix_train=h.npix
)
video.plot(scale='log', drange=1e3, outpath=outdir+'out_frames.png')
video.plot_gif(scale='log', drange=1e3, outpath=outdir+'out.gif')
if spectral:
    video.plot_spectral_index(outpath=outdir+'out_alpha.png')
video.save_h5(outdir+'video.h5')
