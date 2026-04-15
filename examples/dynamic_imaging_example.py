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

"""Dynamic imaging with `kine`.

This file is a comprehensive example on how `kine` can be used to
reconstruct a Stokes I video from intra-day variable EHT-like data.

Here `kine` employs two Neural Fields to decompose the source into a
dynamic and a static component, whose addition represents the full solution.

The imaging process has 3 steps:
    - Step 0: Find the static component flux density.
    - Step 1: Reconstruct video from a disk initialization.
    - Step 2: Final reconstruction initializing to prev. video.

References:
    [1] Fuentes, A., Foschi, M. et al., "Validation of horizon-scale
    Sagittarius A* video reconstruction with kine" In prep. (2026).
"""

import queue
import argparse
import warnings
from functools import partial
from collections import OrderedDict as odict

import yaml
import numpy as np
import ehtim as eh
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

print(
    """
-------
Loading data...
    """
)

with ut.no_print():
    # Loading
    obs = ob.Obsdata.load_uvfits(par.obs)

    # Flag scan with no ALMA nor SPT
    obs = obs.flag_UT_range(UT_start_hour=11.83, UT_stop_hour=12.05)

    # Pre-processing
    obs = obs.avg_coherent(h.tavg)
    obs = obs.add_fractional_noise(h.syserr)
    obs = obs.flag_UT_range(
        UT_start_hour=h.tflag['t0'],
        UT_stop_hour=h.tflag['t1'],
        output=h.tflag['out']
    )
    obs = obs.flag_empty()
    obs = obs.norm_to_max()

    # Split data
    obslist = obs.split_obs(min_bl=h.min_bl)

    # Get light curve (or zbl flux density)
    lcurve = obs.get_lightcurve(min_bl=h.min_bl)

# ______________________________________________________________________________
# STEP 0: Find the flux density of the static and dynamic components

print(
    """ \n
-------
STEP 0: FIND STATIC AND DYNAMIC FLUX DENSITIES \n
    """
)

# ______________________________________________________________________________
# Set up grid of input coordinates and data products

# Field of view
fov = h.fov_uas_0 * eh.RADPERUAS

# Set time coordinates
times = ut.list_to_jaxarr([ob.tstart for ob in obslist])
ntimes = len(obslist)

# Set 2+3D coordinate grids
s_grid = ut.get_grid(h.npix_0, h.npix_0)
d_grid = ut.get_grid(h.npix_0, h.npix_0, ntimes, times=times)

# Set polarization channels
outdim = 1
if 'visQ' in h.data_prod: outdim = 4
if 'visV' in h.data_prod: outdim = 5

# Set empty image for image metadata and dimensions
improxy = eh.image.make_square(obs, h.npix_0, fov, pol_prim='I')

# Compute lists of data products (target), uncertainties (sigma),
# and Fourier matrix (A) at each observed time
print(
    """
-------
Retrieving data products...
    """
)

s_data, d_data = {}, {}
for dtype in h.data_prod:
    # All data for static emission
    target, sigma, A = ob.Obsdata.get_data(obs, dtype, improxy)
    s_data[dtype] = {
        'target': target,
        'sigma': sigma,
        'A': A
    }
    # Snapshot data for recovering dynamics
    target, sigma, A, padmask = ob.Obsdata.get_data(obslist, dtype, improxy)
    d_data[dtype] = {
        'target': target,
        'sigma': sigma,
        'A': A,
        'padmask': padmask
    }
    del target, sigma, A, padmask

# ______________________________________________________________________________
# Set up the network and training scheme

# ...................
# Initialize networks

# Static Neural Field
s_network = mo.NeuralField(
    posenc_deg=tuple(h.nposenc[-2:]),
    outdim=outdim,
    depth=h.s_depth,
    width=h.width,
    activ=nn.gelu,
    outactiv=nn.sigmoid,
    outshift=h.outshift,
    scaling_i=h.scaling_i
)
s_batch_stats = s_network.init(
    rkey,
    jnp.ones([s_grid.shape[-1]]),
    train=True
)
s_params = s_network.init(
    rkey,
    jnp.ones([s_grid.shape[-1]]),
    train=True
)

# Schedule and train state
s_sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter_0: 0.01}
)
s_state = tr.Trainer.create(
    apply_fn=s_network.apply,
    params=s_params['params'].unfreeze(),
    batch_stats=s_batch_stats['batch_stats'].unfreeze(),
    tx=optax.adamax(s_sched)
)

# Dynamic Neural Field
d_network = mo.NeuralField(
    posenc_deg=tuple(h.nposenc),
    outdim=outdim,
    depth=h.d_depth,
    width=h.width,
    activ=nn.gelu,
    outactiv=nn.sigmoid,
    outshift=h.outshift,
    scaling_i=h.scaling_i
)
d_batch_stats = d_network.init(
    rkey,
    jnp.ones([d_grid.shape[-1]]),
    train=True
)
d_params = d_network.init(
    rkey,
    jnp.ones([d_grid.shape[-1]]),
    train=True
)

# Schedule and train state
d_sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter_0: 0.1}
)
d_state = tr.Trainer.create(
    apply_fn=d_network.apply,
    params=d_params['params'].unfreeze(),
    batch_stats=d_batch_stats['batch_stats'].unfreeze(),
    tx=optax.adamax(d_sched)
)

# ______________________________________________________________________________
# Initialize the neural field

# ........................................
# Create static and dynamic initialization

# Static initialization
init_img = vi.Video(
    jnp.array([obs.tstart]),
    h.npix_0,
    fov,
    obs.ra,
    obs.dec,
    h.initniter_0
)
init_img.add_tophat(jnp.ones([1]), h.init_params)
init_img.plot()

# Dynamic initialization
init_vid = vi.Video(times, h.npix_0, fov, obs.ra, obs.dec, h.initniter_0)
init_vid.add_tophat(jnp.ones_like(lcurve), h.init_params)
init_vid.plot()

# ..............................
# Static initialization training

# Initialize video and loss
s_init = vi.Video(
    jnp.array([obs.tstart]),
    h.npix_0,
    fov,
    obs.ra,
    obs.dec,
    h.initniter_0
)
lloss, loss = [], 0

# Looping over epochs
print(
    """
-------
Initializing static network
    """
)
for i in (pbar := tqdm(range(1, h.initniter_0+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, _, out, s_state = tr.Trainer.train_step(
        odict(
            state=s_state,
            grid=s_grid,
            init_img=init_img.iarr
        )
    )

    # Save loss
    lloss.append(loss)

    # Save video
    if i == 1 or i % 500 == 0 or i == h.initniter_0:
        q.put(
            dict(
                video=s_init,
                out=out,
                loss=lloss,
                outpath='./s_init_0.png'
            )
        )

# Block queue
q.join()

# ...............................
# Dynamic initialization training

# Initialize video and loss
d_init = vi.Video(times, h.npix_0, fov, obs.ra, obs.dec, h.initniter_0)
lloss, loss = [], 0

# Looping over epochs
print(
    """
-------
Initializing dynamic network
    """
)
for i in (pbar := tqdm(range(1, h.initniter_0+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, _, out, d_state = tr.Trainer.train_step(
        odict(
            state=d_state,
            grid=d_grid,
            init_vid=init_vid.iarr
        )
    )

    # Save loss
    lloss.append(loss)

    # Save video
    if i == 1 or i % 500 == 0 or i == h.initniter_0:
        q.put(
            dict(
                video=d_init,
                out=out,
                loss=lloss,
                outpath='./d_init_0.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Training

# Update NPIX
tr.NPIX = h.npix_0

# .....................................
# Initialize gains learnable parameters

# Set gain fitting variables
sites, nsites, nvis, bl_indx, lower, upper = obs.set_gains_vars(
    obslist,
    h.gains_prior
)

# Set exponential schedule
ag_schedule = ut.Schedule(1e-12, 1e-12, h.niter_0)

# Amplitude gains network
ag_network = mo.AmplitudeGains(
    nsites=nsites,
    ntimes=ntimes,
    lower=lower,
    upper=upper
)
ag_params = ag_network.init(
    rkey,
    jnp.ones((ntimes, nvis, 2), dtype=int),
    jnp.ones((ntimes), dtype=int)
)
ag_state = tr.Trainer.create(
    apply_fn=ag_network.apply,
    params=ag_params['params'].unfreeze(),
    tx=optax.adamax(learning_rate=ag_schedule.exponential)
)

# Phase gains network
pg_network = mo.PhaseGains(nsites=nsites, ntimes=ntimes)
pg_params = pg_network.init(
    rkey,
    jnp.ones((ntimes, nvis, 2), dtype=int),
    jnp.ones((ntimes), dtype=int)
)
pg_state = tr.Trainer.create(
    apply_fn=pg_network.apply,
    params=pg_params['params'].unfreeze(),
    tx=optax.adamax(learning_rate=1)
)

# ..............
# Traininig loop

# Initialize video and loss
video = vi.Video(times, h.npix_0, fov, obs.ra, obs.dec, h.niter_0)
lloss, loss = {dp: [] for dp in h.data_prod} | {'border': [], 'min_dyn': []}, 0

# Looping over epochs
print(
    """
-------
Training
    """
)
for i in (pbar := tqdm(range(1, h.niter_0+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, ldict, s_out, d_out, out, s_state, d_state, \
        ag_state, pg_state = tr.Trainer.train_step(
            odict(
                s_state=s_state,
                d_state=d_state,
                ag_state=ag_state,
                pg_state=pg_state,
                s_grid=s_grid,
                d_grid=d_grid,
                data=d_data,
                bl_indx=bl_indx,
        )
    )

    # Save loss
    for l in lloss.keys():
        lloss[l].append(ldict[l])

    # Save video
    if i == 1 or i % 500 == 0 or i == h.niter_0:
        amp_gains = ag_state.params | {'sites': list(sites.keys())}
        q.put(
            dict(
                video=video,
                out=out,
                loss=lloss,
                s_out=s_out,
                d_out=d_out,
                amp_gains=amp_gains,
                outpath='./out_0.png'
            )
        )

# Block queue
q.join()

# Define light curve minimum flux density (aka static flux density)
min_lcurve = ut.get_static_flux(s_out.sum(), lcurve.min())

# ______________________________________________________________________________
# STEP 1: Run assigning the flux density ratio found in the previous step

print(
    """ \n
-------
STEP 1: DIVIDE FLUX DENSITIES AND RUN \n
    """
)

# ...............................................
# Re-load data products with new FOV and metadata

# Re-set fov
fov = h.fov_uas_1 * eh.RADPERUAS

# Set 2+3D coordinate grids
s_grid = ut.get_grid(h.npix_1, h.npix_1)
d_grid = ut.get_grid(h.npix_1, h.npix_1, ntimes, times=times)

# Set empty image for image metadata and dimensions
improxy = eh.image.make_square(obs, h.npix_1, fov, pol_prim='I')

# Compute lists of data products (target), uncertainties (sigma),
# and Fourier matrices (A) at each observed time
print(
    """
-------
Retrieving data products...
    """
)

s_data, d_data = {}, {}
for dtype in h.data_prod:
    # All data for static emission
    target, sigma, A = ob.Obsdata.get_data(obs, dtype, improxy)
    s_data[dtype] = {
        'target': target,
        'sigma': sigma,
        'A': A
    }
    # Snapshot data for recovering dynamics
    target, sigma, A, padmask = ob.Obsdata.get_data(obslist, dtype, improxy)
    d_data[dtype] = {
        'target': target,
        'sigma': sigma,
        'A': A,
        'padmask':
        padmask
    }
    del target, sigma, A, padmask

# ...................
# Initialize networks

# Static Neural Field
s_batch_stats = s_network.init(
    rkey,
    jnp.ones([s_grid.shape[-1]]), train=True
)
s_params = s_network.init(
    rkey,
    jnp.ones([s_grid.shape[-1]]), train=True
)

# Schedule and train state
s_sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter_1: 0.01}
)
s_state = tr.Trainer.create(
    apply_fn=s_network.apply,
    params=s_params['params'].unfreeze(),
    batch_stats=s_batch_stats['batch_stats'].unfreeze(),
    tx=optax.adamax(s_sched)
)

# Dynamic Neural Field
d_batch_stats = d_network.init(
    rkey,
    jnp.ones([d_grid.shape[-1]]), train=True
)
d_params = d_network.init(
    rkey,
    jnp.ones([d_grid.shape[-1]]), train=True
)

# Schedule and train state
d_sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter_1: 0.1}
)
d_state = tr.Trainer.create(
    apply_fn=d_network.apply,
    params=d_params['params'].unfreeze(),
    batch_stats=d_batch_stats['batch_stats'].unfreeze(),
    tx=optax.adamax(d_sched)
)

# ........................................
# Create static and dynamic initialization

# Static initialization
init_img = vi.Video(
    jnp.array([obs.tstart]),
    h.npix_1,
    fov,
    obs.ra,
    obs.dec,
    h.initniter_1
)
init_img.add_tophat(jnp.ones([1]), h.init_params)
init_img.plot()

# Dynamic initialization
init_vid = vi.Video(times, h.npix_1, fov, obs.ra, obs.dec, h.initniter_1)
init_vid.add_tophat(jnp.ones_like(lcurve), h.init_params)
init_vid.plot()

# ..............................
# Static initialization training

# Initialize video and loss
s_init = vi.Video(
    jnp.array([obs.tstart]),
    h.npix_1,
    fov,
    obs.ra,
    obs.dec,
    h.initniter_1
)
lloss, loss = [], 0

# Looping over epochs
print(
    """
-------
Initializing static network
    """
)
for i in (pbar := tqdm(range(1, h.initniter_1+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, _, out, s_state = tr.Trainer.train_step(
        odict(
            state=s_state,
            grid=s_grid,
            init_img=init_img.iarr
        )
    )

    # Save loss
    lloss.append(loss)

    # Save video
    if i == 1 or i % 500 == 0 or i == h.initniter_1:
        q.put(
            dict(
                video=s_init,
                out=out,
                loss=lloss,
                outpath='./s_init_1.png'
            )
        )

# Block queue
q.join()

# ...............................
# Dynamic initialization training

# Initialize video and loss
d_init = vi.Video(times, h.npix_1, fov, obs.ra, obs.dec, h.initniter_1)
lloss, loss = [], 0

# Looping over epochs
print(
    """
-------
Initializing dynamic network
    """
)
for i in (pbar := tqdm(range(1, h.initniter_1+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, _, out, d_state = tr.Trainer.train_step(
        odict(
            state=d_state,
            grid=d_grid,
            init_vid=init_vid.iarr
        )
    )

    # Save loss
    lloss.append(loss)

    # Save video
    if i == 1 or i % 500 == 0 or i == h.initniter_1:
        q.put(
            dict(
                video=d_init,
                out=out,
                loss=lloss,
                outpath='./d_init_1.png'
            )
        )

# Block queue
q.join()

# .....................................
# Initialize gains learnable parameters

# Set gain fitting variables
sites, nsites, nvis, bl_indx, lower, upper = obs.set_gains_vars(
    obslist,
    h.gains_prior
)

# Set exponential schedule
ag_schedule = ut.Schedule(5e-5, 1e-3, h.niter_1)

# Amplitude gains network
ag_network = mo.AmplitudeGains(
    nsites=nsites,
    ntimes=ntimes,
    lower=lower,
    upper=upper
)
ag_params = ag_network.init(
    rkey,
    jnp.ones((ntimes, nvis, 2), dtype=int),
    jnp.ones((ntimes), dtype=int)
)
ag_state = tr.Trainer.create(
    apply_fn=ag_network.apply,
    params=ag_params['params'].unfreeze(),
    tx=optax.adamax(learning_rate=ag_schedule.exponential)
)

# Phase gains network
pg_network = mo.PhaseGains(nsites=nsites, ntimes=ntimes)
pg_params = pg_network.init(
    rkey,
    jnp.ones((ntimes, nvis, 2), dtype=int),
    jnp.ones((ntimes), dtype=int)
)
pg_state = tr.Trainer.create(
    apply_fn=pg_network.apply,
    params=pg_params['params'].unfreeze(),
    tx=optax.adamax(learning_rate=1)
)

# ..............
# Traininig loop

# Update NPIX
tr.NPIX = h.npix_1

# Initialize video and loss
video = vi.Video(times, h.npix_1, fov, obs.ra, obs.dec, h.niter_1)
lloss = {dp: [] for dp in h.data_prod} \
      | {'border': [], 's_flux': [], 'd_flux': []}
loss = 0

# Looping over epochs
print(
    """
-------
Training
    """
)
for i in (pbar := tqdm(range(1, h.niter_1+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, ldict, s_out, d_out, out, s_state, d_state, \
        ag_state, pg_state = tr.Trainer.train_step(
            odict(
                s_state=s_state,
                d_state=d_state,
                ag_state=ag_state,
                pg_state=pg_state,
                s_grid=s_grid,
                d_grid=d_grid,
                data=d_data,
                bl_indx=bl_indx,
                min_lcurve=min_lcurve,
                lcurve=lcurve
        )
    )

    # Save loss
    for l in lloss.keys():
        lloss[l].append(ldict[l])

    # Save video
    if i == 1 or i % 500 == 0 or i == h.niter_1:
        amp_gains = ag_state.params | {'sites': list(sites.keys())}
        q.put(
            dict(
                video=video,
                out=out,
                loss=lloss,
                s_out=s_out,
                d_out=d_out,
                amp_gains=amp_gains,
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
video.plot_gif(s_out=s_out, d_out=d_out, outpath='./out_1.gif')

# .......................................
# Save video, gains, and model parameters

# Re-set grid up for output video
s_grid = ut.get_grid(h.npix_1, h.npix_1)
d_grid = ut.get_grid(h.npix_1, h.npix_1, ntimes, times=times)

# Generate output video
print(
    """
Sampling network on a finer grid and saving results...
    """
)
video = vi.Video(times, h.npix_1, fov, obs.ra, obs.dec, h.niter_1)
video.from_states(
    s_state,
    d_state,
    s_grid,
    d_grid,
    lcurve,
    min_lcurve,
    amp_gains=amp_gains
)
video.save_h5('./video_1.h5')
video.save_gains('./gains_1.txt')

# ______________________________________________________________________________
# STEP 2: Rerun using as network initilization the previous video

print(
    """ \n
-------
STEP 2: RE-RUN WITH BETTER INITIALIZATION \n
    """
)

# ...............................................
# Re-load data products with new FOV and metadata

# Re-set fov
fov = h.fov_uas_2 * eh.RADPERUAS

# Set 2+3D coordinate grids
s_grid = ut.get_grid(h.npix_2, h.npix_2)
d_grid = ut.get_grid(h.npix_2, h.npix_2, ntimes, times=times)

# Set empty image for image metadata and dimensions
improxy = eh.image.make_square(obs, h.npix_2, fov, pol_prim='I')

# Compute lists of data products (target), uncertainties (sigma),
# and Fourier matrix (A) at each observed time
print(
    """
-------
Retrieving data products...
    """
)

s_data, d_data = {}, {}
for dtype in h.data_prod:
    # All data for static emission
    target, sigma, A = ob.Obsdata.get_data(obs, dtype, improxy)
    s_data[dtype] = {
        'target': target,
        'sigma': sigma,
        'A': A
    }
    # Snapshot data for recovering dynamics
    target, sigma, A, padmask = ob.Obsdata.get_data(obslist, dtype, improxy)
    d_data[dtype] = {
        'target': target,
        'sigma': sigma,
        'A': A,
        'padmask': padmask
    }
    del target, sigma, A, padmask

# ...................
# Initialize networks

# Static Neural Field
s_batch_stats = s_network.init(
    rkey,
    jnp.ones([s_grid.shape[-1]]), train=True
)
s_params = s_network.init(
    rkey,
    jnp.ones([s_grid.shape[-1]]), train=True
)

# Schedule and train state
s_sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter_2: 0.01}
)
s_state = tr.Trainer.create(
    apply_fn=s_network.apply,
    params=s_params['params'].unfreeze(),
    batch_stats=s_batch_stats['batch_stats'].unfreeze(),
    tx=optax.adamax(s_sched)
)

# Change activation
sharpgelu = partial(mo.sharpgelu, s=3)
d_network.activ = sharpgelu

# Dynamic Neural Field
d_batch_stats = d_network.init(
    rkey,
    jnp.ones([d_grid.shape[-1]]), train=True
)
d_params = d_network.init(
    rkey,
    jnp.ones([d_grid.shape[-1]]), train=True
)

# Schedule and train state
d_sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter_2: 0.1}
)
d_state = tr.Trainer.create(
    apply_fn=d_network.apply,
    params=d_params['params'].unfreeze(),
    batch_stats=d_batch_stats['batch_stats'].unfreeze(),
    tx=optax.adamax(d_sched)
)

# ........................................
# Create static and dynamic initialization

# Static initialization
init_img = vi.Video(
    jnp.array([obs.tstart]),
    h.npix_2,
    fov,
    obs.ra,
    obs.dec,
    h.initniter_2
)
init_img.from_h5('./video_1.h5', blur=0, fn=np.median)
init_img.plot()

# Dynamic initialization
init_vid = vi.Video(times, h.npix_2, fov, obs.ra, obs.dec, h.initniter_2)
init_vid.from_h5('./video_1.h5', blur=30, fn=np.median)
init_vid.plot()

# ..............................
# Static initialization training

# Initialize video and loss
s_init = vi.Video(
    jnp.array([obs.tstart]),
    h.npix_2,
    fov,
    obs.ra,
    obs.dec,
    h.initniter_2
)
lloss, loss = [], 0

# Looping over epochs
print(
    """
-------
Initializing static network
    """
)
for i in (pbar := tqdm(range(1, h.initniter_2+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, _, out, s_state = tr.Trainer.train_step(
        odict(
            state=s_state,
            grid=s_grid,
            init_img=init_img.iarr
        )
    )

    # Save loss
    lloss.append(loss)

    # Save video
    if i == 1 or i % 500 == 0 or i == h.initniter_2:
        q.put(
            dict(
                video=s_init,
                out=out,
                loss=lloss,
                outpath='./s_init_2.png'
            )
        )

# Block queue
q.join()

# ...............................
# Dynamic initialization training

# Initialize video and loss
d_init = vi.Video(times, h.npix_2, fov, obs.ra, obs.dec, h.initniter_2)
lloss, loss = [], 0

# Looping over epochs
print(
    """
-------
Initializing dynamic network
    """
)
for i in (pbar := tqdm(range(1, h.initniter_2+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, _, out, d_state = tr.Trainer.train_step(
        odict(
            state=d_state,
            grid=d_grid,
            init_vid=init_vid.iarr
        )
    )
   
    # Save loss
    lloss.append(loss)

    # Save video
    if i == 1 or i % 500 == 0 or i == h.initniter_2:
        q.put(
            dict(
                video=d_init,
                out=out,
                loss=lloss,
                outpath='./d_init_2.png'
            )
        )

# Block queue
q.join()

# .....................................
# Initialize gains learnable parameters

# Set gain fitting variables
sites, nsites, nvis, bl_indx, lower, upper = obs.set_gains_vars(
    obslist,
    h.gains_prior
)

# Set exponential schedule
ag_schedule = ut.Schedule(5e-5, 1e-3, h.niter_2)

# Amplitude gains network
ag_network = mo.AmplitudeGains(nsites=nsites, ntimes=ntimes, lower=lower, upper=upper)
ag_params = ag_network.init(
    rkey,
    jnp.ones((ntimes, nvis, 2), dtype=int),
    jnp.ones((ntimes), dtype=int)
)
ag_state = tr.Trainer.create(
    apply_fn=ag_network.apply,
    params=ag_params['params'].unfreeze(),
    tx=optax.adamax(learning_rate=ag_schedule.exponential)
)

# Phase gains network
pg_network = mo.PhaseGains(nsites=nsites, ntimes=ntimes)
pg_params = pg_network.init(
    rkey,
    jnp.ones((ntimes, nvis, 2), dtype=int),
    jnp.ones((ntimes), dtype=int)
)
pg_state = tr.Trainer.create(
    apply_fn=pg_network.apply,
    params=pg_params['params'].unfreeze(),
    tx=optax.adamax(learning_rate=1)
)

# ..............
# Traininig loop

# Update NPIX
tr.NPIX = h.npix_2

# Initialize video and loss
video = vi.Video(times, h.npix_2, fov, obs.ra, obs.dec, h.niter_2)
lloss = {dp: [] for dp in h.data_prod} \
      | {'border': [], 's_flux': [], 'd_flux': []}
loss = 0

# Looping over epochs
print(
    """
-------
Training
    """
)
for i in (pbar := tqdm(range(1, h.niter_2+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, ldict, s_out, d_out, out, s_state, d_state, \
        ag_state, pg_state = tr.Trainer.train_step(
            odict(
                s_state=s_state,
                d_state=d_state,
                ag_state=ag_state,
                pg_state=pg_state,
                s_grid=s_grid,
                d_grid=d_grid,
                data=d_data,
                bl_indx=bl_indx,
                min_lcurve=min_lcurve,
                lcurve=lcurve,
                w_border=0,
        )
    )

    # Save loss
    for l in lloss.keys():
        lloss[l].append(ldict[l])

    # Save video
    if i == 1 or i % 500 == 0 or i == h.niter_2:
        amp_gains = ag_state.params | {'sites': list(sites.keys())}
        q.put(
            dict(
                video=video,
                out=out,
                loss=lloss,
                s_out=s_out,
                d_out=d_out,
                amp_gains=amp_gains,
                outpath='./out_2.png'
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
video.plot_gif(s_out=s_out, d_out=d_out, outpath='./out_2.gif')

# .......................................
# Save video, gains, and model parameters

# Re-set grid up for output video
s_grid = ut.get_grid(h.npix_2, h.npix_2)
d_grid = ut.get_grid(h.npix_2, h.npix_2, ntimes, times=times)

# Generate output video
print(
    """
Sampling network on a finer grid and saving results...
    """
)
video = vi.Video(times, h.npix_2, fov, obs.ra, obs.dec, h.niter_2)
video.from_states(
    s_state,
    d_state,
    s_grid,
    d_grid,
    lcurve,
    min_lcurve,
    amp_gains=amp_gains
)
video.save_h5('./video_2.h5')
video.save_gains('./gains_2.txt')
