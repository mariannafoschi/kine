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

"""Multi-epoch dynamic imaging with `kine`.

This file is a comprehensive example on how `kine` can be used to
reconstruct a Stokes I video from multiple observations of VLBI arrays.

This has been tested for hundreds of different epochs spanning decades
of source monitoring with programs like MOJAVE, BEAM-ME, or ngEHT (simulated).

References:
    [1] Foschi, M., Zhao, B., Fuentes, A. et al., "Video reconstruction of
    variable interferometric observations with neural fields." Under rev. (2026)
"""

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
import kine.video as vi
import kine.model as mo
import kine.obsdata as ob
import kine.trainer as tr

# Filter warnings
warnings.filterwarnings('ignore')

# Initialize async worker
q = queue.Queue()
ut.init_worker(vi.Video.async_plot, q)

# ______________________________________________________________________________
# Load hyperparameters and set RNG keys

# Load arguments from command line
par = argparse.ArgumentParser()
par.add_argument('-obs', type=str, help='uvfits file name')
par.add_argument('-yml', type=str, help='hyperparameter file name')
par = par.parse_args()

# Load hyperparameters from YAML file
with open(par.yml, 'r') as f:
    h = yaml.safe_load(f)
h = ut.HyperParams(h)

# Set RNG keys
rkey = jax.random.PRNGKey(h.seed)
np.random.seed(h.seed)

# ______________________________________________________________________________
# Load and pre-process data

obslist, bad = [], []
obspath = sorted(glob.glob(par.obs + '*.uvfits'))

print(
    """
-------
Loading multi-epoch data...
    """
)

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

print(
    f"""
Total Good Files: {len(obspath) - len(bad)}
Total Bad Files: {len(bad)} \n
    """
)

# Fix metadata and empty array scans (hacky fix)
for i, _ in enumerate(obslist):
    obslist[i].fix_multiepoch(obslist[-1])
    if len(obslist[i].scans) == 0:
        obslist[i].scans = None

# Get light curve (or zbl flux density)
lcurve = jnp.array([ob.get_zbl() for ob in obslist])

# ______________________________________________________________________________
# Set up grid of input coordinates and data products

# Field of view
fov = h.fov_uas * eh.RADPERUAS

# Set time coordinates
times = ut.get_times_multiepoch(obspath)
dates = ut.get_times_multiepoch(obspath, ymd=True)
ntimes = len(obslist)

# Set 3D coordinate grids
grid = ut.get_grid(h.npix_1, h.npix_1, ntimes, times=times)

# Set polarization channels
outdim = 1
if 'visQ' in h.data_prod: outdim = 4
if 'visV' in h.data_prod: outdim = 5

# Set empty image for image metadata and dimensions
improxy = eh.image.make_square(obslist[0], h.npix_1, fov, pol_prim='I')

# Compute lists of data products (target), uncertainties (sigma),
# and Fourier matrix (A) at each observed time
print(
    """
-------
Retrieving data products from each epoch...
    """
)

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

print(
    """
-------
Setting up NUFFT variables...
    """
)

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

# Neural Field
network = mo.NeuralField(
    posenc_deg=tuple(h.nposenc),
    outdim=outdim,
    depth=h.depth,
    width=h.width,
    activ=nn.gelu,
    outactiv=nn.softplus,
    outshift=h.outshift,
    scaling_i=h.scaling_i
)
batch_stats = network.init(
    rkey,
    jnp.ones([grid.shape[-1]]), train=True
)
params = network.init(
    rkey,
    jnp.ones([grid.shape[-1]]), train=True
)

# Schedule and train state
sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter: 0.1}
)
state = tr.Trainer.create(
    apply_fn=network.apply,
    params=params['params'].unfreeze(),
    batch_stats=batch_stats['batch_stats'].unfreeze(),
    tx=optax.adamax(sched)
)

# ______________________________________________________________________________
# Initialize the neural field

# .....................
# Create initialization

init_vid = vi.Video(
    times,
    h.npix_1,
    fov,
    obs.ra,
    obs.dec,
    h.initniter,
    dates=dates
)
init_vid.add_tophat(lcurve, h.init_params)
init_vid.plot()

# .......................
# Initialization training

# Initialize video and loss
init = vi.Video(
    times,
    h.npix_1,
    fov,
    obs.ra,
    obs.dec,
    h.initniter,
    dates=dates
)
lloss, loss = [], 0

# Looping over epochs
print(
    """
-------
Initializing network
    """
)
for i in (pbar := tqdm(range(1, h.initniter+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, _, out, state = tr.Trainer.train_step(
        odict(
            state=state,
            grid=grid,
            init_vid=init_vid.iarr
        )
    )
   
    # Save loss
    lloss.append(loss)

    # Save video
    if i == 1 or i % 500 == 0 or i == h.initniter:
        q.put(
            dict(
                video=init,
                out=out,
                loss=lloss,
                scale='log',
                drange=5e2,
                outpath='./init_1.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Training

# Update NPIX for training with NUFFT
tr.NPIX = h.npix_1

# ..............
# Traininig loop

# Initialize video and loss
video = vi.Video(times, h.npix_1, fov, obs.ra, obs.dec, h.niter, dates=dates)
lloss, loss = {dp: [] for dp in h.data_prod} | {'lcurve': []}, 0

# Looping over epochs
print(
    """
-------
Training
    """
)
for i in (pbar := tqdm(range(1, h.niter+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, ldict, out, state = tr.Trainer.train_step(
        odict(
            state=state,
            grid=grid,
            data=data,
            lcurve=lcurve,
            uvpoints=uv,
            pulsefac=pulse,
            uvind=uvind,
            triangles=tria,
            quadrangles=quad
        )
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
                scale='log',
                drange=5e3,
                outpath='./out_1.png'
            )
        )

# Block queue
q.join()

# Visualize
print(
    """
-------
Saving results...
    """
)
video.from_video(out, loss=lloss)
video.plot_gif(scale='log', drange=5e2, outpath='./out_1.gif')

# ______________________________________________________________________________
# Save video, gains, and model parameters

# Re-set grid up for output video
grid = ut.get_grid(h.npix_2, h.npix_2, ntimes, times=times)

# Generate output video
print(
    """
Sampling network on a finer grid and saving results...
    """
)
video = vi.Video(times, h.npix_2, fov, obs.ra, obs.dec, h.niter, dates=dates)
video.from_state(state, grid)
video.save_h5('./video_1.h5')
