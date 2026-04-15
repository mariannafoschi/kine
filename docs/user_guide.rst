==========
User Guide
==========

This guide explains the ``kine`` imaging pipeline in detail. For the mathematical background, see Foschi, Zhao, Fuentes et al. (2026) and Fuentes, Foschi et al. (2026).

Overview
--------

``kine`` reconstructs time-variable radio images by training a *Neural Field* --- a coordinate-based neural network that maps space-time coordinates ``(x, y, t)`` to brightness values. The network is optimized so that the Fourier transform of its output matches the observed interferometric data products (visibilities, closure phases, etc.).

The typical imaging pipeline has three stages, each at progressively higher resolution:

1. **Step 0** --- Find the static component flux density at low resolution.
2. **Step 1** --- Reconstruct the video using disk initialization with the found flux ratio.
3. **Step 2** --- Refine at higher resolution, initializing from the Step 1 result.

Each step itself consists of:

- Network initialization (pre-training to match a simple geometric model)
- Data-driven training (optimizing against interferometric data products)


Data Loading and Preprocessing
------------------------------

``kine`` uses the :class:`kine.obsdata.Obsdata` class, which extends ``ehtim``'s ``Obsdata`` with additional methods. Load data from UV-FITS files:

.. code-block:: python

   import kine.obsdata as ob

   obs = ob.Obsdata.load_uvfits('observation.uvfits')

**Common preprocessing steps:**

.. code-block:: python

   # Time-average to reduce data volume
   obs = obs.avg_coherent(60)  # 60 seconds

   # Add systematic noise budget
   obs = obs.add_fractional_noise(0.01)  # 1%

   # Keep only a specific time window
   obs = obs.flag_UT_range(
       UT_start_hour=10.85,
       UT_stop_hour=14.05,
       output='flagged'
   )

   # Remove telescopes with no data
   obs = obs.flag_empty()

   # Normalize amplitudes (required for static+dynamic decomposition)
   obs = obs.norm_to_max()

**Split into time snapshots** for dynamic imaging:

.. code-block:: python

   obslist = obs.split_obs(min_bl=4)  # minimum 4 baselines per snapshot

**Extract the light curve** for flux density constraints:

.. code-block:: python

   lcurve = obs.get_lightcurve(min_bl=4)


Coordinate Grids
-----------------

The neural field is trained on a grid of space-time coordinates. Use :func:`kine.utils.get_grid` to generate these:

.. code-block:: python

   import kine.utils as ut

   # 2D grid for static imaging (x, y)
   s_grid = ut.get_grid(npix, npix)

   # 3D grid for dynamic imaging (t, x, y)
   times = ut.list_to_jaxarr([o.tstart for o in obslist])
   d_grid = ut.get_grid(npix, npix, len(obslist), times=times)

The ``tdil`` parameter (default: 10) controls time dilation --- a scaling factor that adjusts how much the network weighs temporal vs. spatial variation. Times are normalized to ``[0, 1/tdil]`` while spatial coordinates span ``[0, 1]``.


Data Products
-------------

``kine`` supports multiple interferometric data products, each identified by a string code:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Code
     - Description
     - When to use
   * - ``visI``, ``visQ``, ``visU``, ``visV``
     - Complex visibilities (Stokes I/Q/U/V)
     - When amplitude and phase calibration are reliable
   * - ``ampI``
     - Visibility amplitudes
     - When phase calibration is uncertain
   * - ``logampI``
     - Log visibility amplitudes
     - Preferred for amplitude fitting; better behaved gradients
   * - ``cphaseI``
     - Closure phases
     - Robust to station-based gain errors
   * - ``logcampI``
     - Log closure amplitudes
     - Robust to station-based gain errors
   * - ``bsI``
     - Bispectra
     - Alternative to closure phases
   * - ``mbreve``
     - Complex linear polarization ratio (Q + iU) / I
     - For polarimetric imaging

Data products are computed from the observation using ``ehtim``'s Fourier transform routines:

.. code-block:: python

   import ehtim as eh

   improxy = eh.image.make_square(obs, npix, fov, pol_prim='I')

   # For static imaging (single observation)
   target, sigma, A = ob.Obsdata.get_data(obs, 'logampI', improxy)

   # For dynamic imaging (list of snapshots)
   target, sigma, A, padmask = ob.Obsdata.get_data(
       obslist, 'logampI', improxy
   )

The returned arrays are:

- ``target``: measured data values
- ``sigma``: measurement uncertainties
- ``A``: Discrete Fourier Transform matrices mapping image pixels to visibilities
- ``padmask``: binary mask for handling variable-length snapshot data


Neural Field Architecture
-------------------------

The core model is :class:`kine.model.NeuralField`, an MLP that predicts full-polarimetric emission from coordinates:

.. code-block:: python

   import kine.model as mo
   from flax import linen as nn

   network = mo.NeuralField(
       posenc_deg=(6, 0, 0),  # positional encoding degrees for (t, x, y)
       outdim=1,               # 1=Stokes I only, 4=I+Q+U, 5=I+Q+U+V
       depth=6,                # hidden layers
       width=256,              # neurons per layer
       activ=nn.gelu,          # hidden activation
       outactiv=nn.softplus,   # output activation for Stokes I
       outshift=10,            # shift before output activation
       scaling_i=1.0,          # Stokes I output scaling
   )

**Positional encoding** is critical for capturing fine spatial and temporal structure. The ``posenc_deg`` tuple sets the number of Fourier feature frequencies for each input dimension ``(t, x, y)``. Higher values enable sharper features but may cause overfitting. Typical values:

- Temporal: 4--6 (captures time variability)
- Spatial: 0 (the network learns spatial structure from the MLP itself)

**Output channels** depend on ``outdim``:

- ``outdim=1``: Stokes I only
- ``outdim=4``: Stokes I, linear polarization fraction (ml), sin(2 xi), cos(2 xi)
- ``outdim=5``: Above + circular polarization fraction (mc)

For **linear polarimetric imaging** with a fixed Stokes I, use :class:`kine.model.NeuralFieldPol` instead, which outputs only the polarization parameters (ml, sin(2 xi), cos(2 xi)).


Static + Dynamic Decomposition
-------------------------------

For sources with both persistent and time-variable structure (e.g., Sgr A*), ``kine`` can decompose the emission into static and dynamic components using two separate neural fields:

.. math::

   I(x, y, t) = S_\mathrm{static} \cdot f_\mathrm{static}(x, y) + S_\mathrm{dynamic}(t) \cdot f_\mathrm{dynamic}(x, y, t)

where:

- :math:`f_\mathrm{static}` is a 2D neural field (time-independent)
- :math:`f_\mathrm{dynamic}` is a 3D neural field (time-dependent)
- :math:`S_\mathrm{static}` is the static flux density (found in Step 0)
- :math:`S_\mathrm{dynamic}(t) = \mathrm{lightcurve}(t) - S_\mathrm{static}`

Both networks are normalized so their outputs sum to unity, and the actual flux is assigned through the light curve.

**Step 0** finds the static flux by running a short training with a regularizer that minimizes persistent flux in the dynamic component. The result is extracted via :func:`kine.utils.get_static_flux`.


Gain Fitting
------------

Telescope amplitude and phase gains can be fit simultaneously with the image using learnable parameters:

.. code-block:: python

   # Set up gain variables
   gains_prior = {
       'AA': [0.97, 1.03],  # ALMA: tight bounds
       'LM': [0.85, 1.15],  # LMT: looser bounds
       # ... per-telescope bounds
   }
   sites, nsites, nvis, bl_indx, lower, upper = obs.set_gains_vars(
       obslist, gains_prior
   )

   # Create amplitude gains model
   ag_network = mo.AmplitudeGains(
       nsites=nsites, ntimes=len(obslist),
       lower=lower, upper=upper
   )

   # Create phase gains model
   pg_network = mo.PhaseGains(
       nsites=nsites, ntimes=len(obslist)
   )

Gain bounds (``gains_prior``) are specified per telescope as ``[lower, upper]`` multiplicative factors. Well-calibrated stations (e.g., ALMA) should have tight bounds; stations with known calibration issues should have looser bounds.

Gains are clipped to their allowed ranges during training:

- **Amplitude gains**: clipped to ``[lower, upper]`` per telescope
- **Phase gains**: wrapped to ``[-pi, pi]``


Training
--------

Training uses the :class:`kine.trainer.Trainer` class, which extends Flax's ``TrainState`` with batch normalization support and loss computation.

**Create a training state:**

.. code-block:: python

   import optax
   import kine.trainer as tr

   state = tr.Trainer.create(
       apply_fn=network.apply,
       params=params['params'],
       batch_stats=params['batch_stats'],
       tx=optax.adamax(1e-3)
   )

**The training step** is a single JIT-compiled function:

.. code-block:: python

   from collections import OrderedDict as odict

   tr.NPIX = npix  # required global for NUFFT and border regularization

   loss, ldict, out, state = tr.Trainer.train_step(
       odict(
           state=state,
           grid=grid,
           data=data,
           lcurve=lcurve
       )
   )

The ``train_step`` method accepts an ``OrderedDict`` (to preserve argument order under ``@jax.jit``) and automatically:

1. Selects the appropriate loss function based on the provided arguments
2. Computes gradients via ``jax.value_and_grad``
3. Applies gradient updates
4. Updates batch normalization statistics

**Loss function selection** is automatic based on keyword arguments:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Arguments present
     - Loss function used
   * - ``init_img`` (2D grid)
     - Static initialization
   * - ``init_vid`` (3D grid)
     - Dynamic initialization
   * - ``init_vid_i``
     - Polarimetric training (fixed Stokes I)
   * - ``s_grid`` (no ``min_lcurve``)
     - Static+dynamic with regularization (Step 0)
   * - ``min_lcurve``
     - Static+dynamic with flux division (Steps 1--2)
   * - ``uvpoints``
     - NUFFT-based computation
   * - Default (3D grid)
     - Single dynamic network

**Learning rate schedules:**

.. code-block:: python

   # Piecewise constant (from optax)
   sched = optax.piecewise_constant_schedule(
       init_value=1e-2,
       boundaries_and_scales={3000: 0.01}
   )

   # Exponential decay (from kine)
   sched = ut.Schedule(lr_i=5e-5, lr_f=1e-3, niter=10000)
   # Use as: optax.adamax(learning_rate=sched.exponential)


Network Initialization
-----------------------

Before data-driven training, the neural field is initialized to reproduce a simple geometric model (typically a uniform disk). This prevents the network from getting stuck in poor local minima.

.. code-block:: python

   # Create a disk initialization target
   init_vid = vi.Video(times, npix, fov, obs.ra, obs.dec, niter)
   init_vid.add_tophat(
       jnp.ones_like(lcurve),
       {'fwhm': 80, 'blur': 20, 'posx': 0, 'posy': 0}
   )

   # Train to match the disk
   for i in range(3000):
       loss, _, out, state = tr.Trainer.train_step(
           odict(state=state, grid=grid, init_vid=init_vid.iarr)
       )

For Step 2, initialization uses the video from the previous step instead of a disk:

.. code-block:: python

   init_vid.from_h5('./video_1.h5', blur=30, fn=np.median)


NUFFT Mode
----------

For large images, ``kine`` supports GPU-based Non-Uniform Fast Fourier Transforms via ``jax-finufft``, which avoids storing large DFT matrices in memory:

.. code-block:: python

   # Get UV coordinates scaled for NUFFT
   psize = fov / npix
   uv = obs.get_uvpoints(psize)
   pulsefac = obs.get_pulsefac(uv, eh.RADPERUAS)

   # Get baseline and closure indices
   blname = obs.get_baselines_nfft()
   triangles = obs.get_closure_indices(blname, 'triangles')
   quadrangles = obs.get_closure_indices(blname, 'quadrangles')

   # Use NUFFT data products
   target, sigma, padmask = ob.Obsdata.get_data_nfft(
       obslist, 'logampI', improxy
   )

Pass the NUFFT variables to ``train_step``:

.. code-block:: python

   tr.Trainer.train_step(odict(
       state=state, grid=grid, data=data,
       lcurve=lcurve, uvpoints=uv,
       pulsefac=pulsefac, uvind=uvind,
       triangles=triangles, quadrangles=quadrangles
   ))


Multi-Epoch Imaging
-------------------

``kine`` can reconstruct source evolution across multiple observation epochs spanning days to years. Each epoch is a separate UV-FITS file.

.. code-block:: python

   import glob

   # Load all epochs
   inpaths = sorted(glob.glob('data/*.uvfits'))
   times = ut.get_times_multiepoch(inpaths)       # MJD times for training
   dates = ut.get_times_multiepoch(inpaths, ymd=True)  # YYYY-MM-DD for labels

   # Load and merge observations
   obslist = []
   for path in inpaths:
       o = ob.Obsdata.load_uvfits(path)
       o.fix_multiepoch(obslist[0])  # match metadata to reference
       obslist.append(o)

A single neural field is trained across all epochs, learning a smooth interpolation of source structure over time. The ``dates`` array is passed to the ``Video`` object for proper labeling in plots and GIFs.


Visualization and Output
-------------------------

The :class:`kine.video.Video` class handles all visualization and export:

**Create a video from the trained network:**

.. code-block:: python

   video = vi.Video(times, npix, fov, obs.ra, obs.dec, niter)

   # From a single network
   video.from_state(state, grid)

   # From static + dynamic decomposition
   video.from_states(s_state, d_state, s_grid, d_grid, lcurve, min_lcurve)

   # From raw output arrays
   video.from_video(out, loss=loss_dict)

**Plot diagnostic frames:**

.. code-block:: python

   # Single network output
   video.plot(outpath='result.png')

   # With static + dynamic components
   video.plot(s_out=s_out, d_out=d_out, outpath='result.png')

   # Logarithmic scale with dynamic range
   video.plot(scale='log', drange=1e3)

**Create animated GIFs:**

.. code-block:: python

   video.plot_gif(outpath='result.gif')
   video.plot_gif(s_out=s_out, d_out=d_out, outpath='result.gif')

**Export results:**

.. code-block:: python

   video.save_h5('video.h5')       # HDF5 video (ehtim-compatible)
   video.save_fits('image.fits')   # FITS image (single frame)
   video.save_gains('gains.txt')   # Fitted amplitude gains

**Asynchronous plotting** during training avoids blocking GPU computation:

.. code-block:: python

   import queue

   q = queue.Queue()
   ut.init_worker(vi.Video.async_plot, q)

   # During training loop:
   q.put(dict(video=video, out=out, loss=loss, outpath='tmp.png'))

   # Wait for all plots to finish
   q.join()


Polarimetric Imaging
--------------------

For linear polarimetric reconstruction, ``kine`` uses a two-stage approach:

1. **Reconstruct Stokes I** using the standard dynamic imaging pipeline.
2. **Reconstruct polarization** with a ``NeuralFieldPol`` that takes the Stokes I video as fixed input.

.. code-block:: python

   # Load the previously reconstructed Stokes I
   pol_video = vi.Video(times, npix, fov, obs.ra, obs.dec, niter)
   pol_video.add_video_i('stokes_i_video.h5')

   # Create polarization network
   pol_network = mo.NeuralFieldPol(
       posenc_deg=(4, 0, 0),
       depth=4, width=256,
       scaling_ml=0.75  # max linear pol. fraction
   )

   # Use visibility Q and U data products
   data = {}
   for dtype in ['visQ', 'visU']:
       target, sigma, A, padmask = ob.Obsdata.get_data(
           obslist, dtype, improxy
       )
       data[dtype] = {
           'target': target, 'sigma': sigma,
           'A': A, 'padmask': padmask
       }

   # Train with fixed Stokes I
   loss, ldict, out, state = tr.Trainer.train_step(
       odict(state=state, grid=grid,
             data=data, init_vid_i=pol_video.iarr)
   )

The polarization network outputs:

- Linear polarization fraction (ml), scaled by ``scaling_ml``
- sin(2 xi) and cos(2 xi), from which the EVPA is computed as xi = 0.5 * arctan2(sin, cos)

Stokes Q and U are then computed as:

.. math::

   Q = -I \cdot m_l \cdot \sin(2\xi), \quad U = I \cdot m_l \cdot \cos(2\xi)
