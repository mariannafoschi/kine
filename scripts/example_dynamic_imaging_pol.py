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

"""Dynamic polarimetric imaging with `kine`.

This file is a comprehensive example on how `kine` can be used to
reconstruct a lin. pol. video from intra-day variable EHT-like data.

Here `kine` employs a Neural Field to recover the lin. pol. fraction
and the EVPA having Stokes I fixed from a prev. Stokes I reconstruction.

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
    # Flag JCMT (because of single pol)
    obs = obs.flag_sites(['JC'])

    # Pre-processing
    obs = obs.avg_coherent(h.tavg)
    obs = obs.add_fractional_noise(h.syserr)
    obs = obs.flag_UT_range(
        UT_start_hour=h.tflag['t0'],
        UT_stop_hour=h.tflag['t1'],
        output=h.tflag['out']
    )
    obs = obs.flag_empty()

    # Split data
    obslist = obs.split_obs(min_bl=h.min_bl)

    # Get light curve (or zbl flux density)
    lcurve = obs.get_lightcurve(min_bl=h.min_bl)

# ______________________________________________________________________________
# Set up grid of input coordinates and data products

# Field of view
fov = h.fov_uas * eh.RADPERUAS

# Set time coordinates
times = jnp.array([ob.tstart for ob in obslist])
ntimes = len(obslist)

# Set 2+3D coordinate grids
grid = ut.get_grid(h.npix, h.npix, ntimes, times=times)

# Set empty image for image metadata and dimensions
improxy = eh.image.make_square(obs, h.npix, fov, pol_prim='I')

# Compute lists of data products (target), uncertainties (sigma),
# and Fourier matrix (A) at each observed time
print(
    """
-------
Retrieving data products...
    """
)

data = {}
for dtype in h.data_prod:
    # Snapshot data for recovering dynamics
    target, sigma, A, padmask = ob.Obsdata.get_data(obslist, dtype, improxy)
    data[dtype] = {
        'target': target,
        'sigma': sigma,
        'A': A,
        'padmask': padmask
    }
    del target, sigma, A, padmask

# ______________________________________________________________________________
# Set up the network and training scheme

# ..................
# Initialize network

# Set activation function
sharpgelu = partial(mo.sharpgelu, s=3)

# Dynamic Neural Field
network = mo.NeuralFieldPol(
    posenc_deg=tuple(h.nposenc),
    outdim=3, # (ml, sin2xi, cos2xi)
    depth=h.depth,
    width=h.width,
    activ=sharpgelu,
    outactiv=nn.sigmoid,
    outshift=h.outshift,
    scaling_ml=h.scaling_ml
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

# .............................
# Create dynamic initialization

# Dynamic initialization
init_vid = vi.Video(times, h.npix, fov, obs.ra, obs.dec, h.initniter)
init_vid.add_video_i(par.obs.replace('uvfits', 'hdf5'))
init_vid.add_constant_linpol()
init_vid.plot()

# ...............................
# Dynamic initialization training

# Initialize video and loss
init = vi.Video(times, h.npix, fov, obs.ra, obs.dec, h.initniter)
init.iarr = init_vid.iarr.copy()
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
            init_vid_ml=init_vid.larr,
            init_vid_x=init_vid.xarr
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
                outpath='./init_pol.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Training

# ..............
# Traininig loop

# Initialize video and loss
video = vi.Video(times, h.npix, fov, obs.ra, obs.dec, h.niter)
video.iarr = init_vid.iarr.copy()
lloss, loss = {dp: [] for dp in h.data_prod} | {'overlap': []}, 0

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
            init_vid_i=init_vid.iarr
        )
    )

    # Save loss
    for l in lloss.keys():
        lloss[l].append(ldict[l])

    # Save video
    if i == 1 or i % 500 == 0 or i == h.niter:
        q.put(
            dict(
                video=video,
                out=out,
                loss=lloss,
                outpath='./out_pol.png'
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
video.plot_gif(outpath='./out_pol.gif')
video.save_h5('./video_pol.h5')
