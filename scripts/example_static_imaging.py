# Copyright (C) 2026 Marianna Foschi, Antonio Fuentes, Brandon Zhao

# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Static imaging with `kine`.

This file is a comprehensive example on how `kine` can be used to reconstruct a 
Stokes I image from one observations of VLBI arrays.

References:
    [1] Foschi, M., Zhao, B., Fuentes, A. et al., "Video reconstruction of
    variable VLBI observations with neural fields." Under rev. (2026)
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
ut.init_worker(vi.Image.async_plot, q)

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

print("-------\nLoading observed data...\n")

try:
    with ut.no_print():
        obs = ob.Obsdata.load_uvfits(par.obs)
        obs = obs.avg_coherent(h.tavg)
        obs = obs.add_fractional_noise(h.syserr)
except Exception as e:
    print(f'Could not load {path}: {e} \n')

# Get light curve (or zbl flux density)
totflux = obs.get_zbl()

# ______________________________________________________________________________
# Set up grid of input coordinates and data products

# Field of view
fov = h.fov_uas * eh.RADPERUAS

# Set 2D coordinate grids
grid = ut.get_grid(h.npix, h.npix)

# Set polarization channels
outdim = 1
if 'visQ' in h.data_prod: outdim = 4
if 'visV' in h.data_prod: outdim = 5

# Set empty image for image metadata and dimensions
improxy = eh.image.make_square(obs, h.npix, fov, pol_prim='I')

# Compute lists of data products (target), uncertainties (sigma),
# and Fourier matrix (A) at each observed time

print("-------\nRetrieving data products from each epoch...\n")

data = {}
for dtype in h.data_prod:
    target, sigma, A = ob.Obsdata.get_data(obs, dtype, improxy)
    data[dtype] = {
        'target': target,
        'sigma': sigma,
        'A': A
    }
    del target, sigma, A

# ______________________________________________________________________________
# Set up the network and optimizer

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
params = network.init(rkey, jnp.ones([grid.shape[-1]]), train=True)
batch_stats = network.init(rkey, jnp.ones([grid.shape[-1]]), train=True)

# Schedule and train state
sched = optax.piecewise_constant_schedule(
    init_value=1e-2,
    boundaries_and_scales={h.initniter: 0.1}
)
state = tr.Trainer.create(
    apply_fn=network.apply,
    params=params['params'],
    batch_stats=batch_stats['batch_stats'],
    tx=optax.adamax(sched)
)

# ______________________________________________________________________________
# Initialize the neural field

print("-------\nInitializing network\n")

# ...........................
# Create initialization image

init_im = vi.Image(
    h.npix,
    fov,
    obs.ra,
    obs.dec,
    h.initniter
)
init_im.add_tophat(totflux, h.init_params)
init_im.plot(outpath='./init_image.png')

# .......................
# Initialization training

# Initialize image and loss
init = vi.Image(
    h.npix,
    fov,
    obs.ra,
    obs.dec,
    h.initniter
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
            init_img=init_im.iarr
        )
    )
   
    # Save loss
    lloss.append(loss)

    # Save image
    if i == 1 or i % 500 == 0 or i == h.initniter:
        q.put(
            dict(
                image=init,
                out=out,
                loss=lloss,
                outpath='./output_init.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Training

print("-------\nTraining\n")

# ..............
# Traininig loop

# Initialize image and loss
image = vi.Image(h.npix, fov, obs.ra, obs.dec, h.niter)
lloss, loss = {dp: [] for dp in h.data_prod} | {'lcurve': []}, 0

# Looping over epochs
for i in (pbar := tqdm(range(1, h.niter+1))):
    pbar.set_description(f'Loss {loss:.1e}')

    # Apply one training step
    loss, ldict, out, state = tr.Trainer.train_step(
        odict(
            state=state,
            grid=grid,
            data=data,
            lcurve=jnp.array([totflux])
        )
    )

    # Save loss
    for l in lloss:
        lloss[l].append(ldict[l])

    # Save image
    if i == 1 or i % 500 == 0 or i == h.niter:
        q.put(
            dict(
                image=image,
                out=out,
                loss=lloss,
                scale='lin',
                outpath='./output_train.png'
            )
        )

# Block queue
q.join()

# ______________________________________________________________________________
# Save image and model parameters

print("-------\nSaving results...\n")

# Sample output image on a finer grid
grid_out = ut.get_grid(h.npix_out, h.npix_out)

# Generate output image
image = vi.Image(h.npix_out, fov, obs.ra, obs.dec, h.niter)
image.from_state(state, grid_out)
image.plot(outpath='./output_image.png')
image.save_fits('./output_image.fits')
