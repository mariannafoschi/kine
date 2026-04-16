===============
Getting Started
===============

Installation
------------

``kine`` relies on JAX for GPU computations and requires a careful installation of CUDA-related packages. A working conda environment specification is provided in ``environment.yml`` at the repository root.

**Step 1: Create the conda environment**

.. code-block:: bash

   conda env create -f environment.yml
   conda activate kine

The environment includes all required dependencies:

- **Core ML stack**: JAX (0.4.23), Flax (0.6.10), Optax (0.2.2)
- **GPU support**: CUDA 11.8
- **Radio astronomy**: ehtim (1.2.4), ehtplot (0.9.0)
- **NUFFT** (optional): finufft (2.3.1), jax-finufft
- **Scientific stack**: NumPy, SciPy, Astropy, Matplotlib

**Step 2: Install kine**

From the repository root directory:

.. code-block:: bash

   pip install -e .


Quick Start
-----------

The fastest way to run ``kine`` is using the provided example scripts in the ``examples/`` folder.

**Run a Stokes I dynamic imaging reconstruction:**

.. code-block:: bash

   cd examples/
   python dynamic_imaging_example.py \
       -obs ../data/dataset.uvfits \
       -yml ./dynamic_imaging_params.yml

Or use the provided bash wrapper:

.. code-block:: bash

   bash run_kine.sh

**Input requirements:**

- A UV-FITS file containing the interferometric observation.
- A YAML parameter file specifying imaging settings (see :doc:`parameters`).

**Output:**

- Diagnostic PNG plots at each stage showing reconstructed frames and loss curves.
- Animated GIF of the reconstructed video.
- HDF5 file containing the full video (loadable with ``ehtim``).
- Text file with fitted telescope gains.


Minimal Example
---------------

Below is a minimal Python script that demonstrates the core ``kine`` workflow. For complete working examples, see the scripts in ``examples/``.

.. code-block:: python

   import jax
   import optax
   import ehtim as eh
   from flax import linen as nn
   from jax import numpy as jnp
   from collections import OrderedDict as odict

   import kine.model as mo
   import kine.obsdata as ob
   import kine.trainer as tr
   import kine.utils as ut
   import kine.video as vi

   # 1. Load and preprocess data
   obs = ob.Obsdata.load_uvfits('observation.uvfits')
   obs = obs.avg_coherent(60)            # 60s time averaging
   obs = obs.add_fractional_noise(0.01)  # 1% systematic error
   obs = obs.flag_empty()
   obslist = obs.split_obs(min_bl=4)

   # 2. Set up coordinate grids
   npix, fov = 32, 160 * eh.RADPERUAS
   times = ut.list_to_jaxarr([o.tstart for o in obslist])
   grid = ut.get_grid(npix, npix, len(obslist), times=times)

   # 3. Compute data products
   improxy = eh.image.make_square(obs, npix, fov, pol_prim='I')
   target, sigma, A, padmask = ob.Obsdata.get_data(
       obslist, 'logampI', improxy
   )
   data = {'logampI': {
       'target': target, 'sigma': sigma,
       'A': A, 'padmask': padmask
   }}

   # 4. Create neural field and training state
   rkey = jax.random.PRNGKey(42)
   network = mo.NeuralField(
       posenc_deg=(6, 0, 0), depth=6, width=256
   )
   params = network.init(rkey, jnp.ones([3]), train=True)
   state = tr.Trainer.create(
       apply_fn=network.apply,
       params=params['params'],
       batch_stats=params['batch_stats'],
       tx=optax.adamax(1e-3)
   )

   # 5. Train
   tr.NPIX = npix
   lcurve = obs.get_lightcurve(min_bl=4)
   for i in range(5000):
       loss, ldict, out, state = tr.Trainer.train_step(
           odict(state=state, grid=grid,
                 data=data, lcurve=lcurve)
       )

   # 6. Visualize
   video = vi.Video(times, npix, fov, obs.ra, obs.dec, 5000)
   video.from_video(out)
   video.plot_gif(outpath='result.gif')
   video.save_h5('result.h5')
