==========
User Guide
==========

This guide walks through the structure of a typical ``kine`` imaging script,
explaining in detail the available options and how to adapt the general 
structure to a specific imaging scenario. 

For the mathematical background of the algorithm and its validation see the
reference papers listed in :doc:`index`, for a list and description of the 
parameters to be used in the YAML file see :doc:`parameters`, for function 
signatures see :doc:`api/index`.


Overview
--------

``kine`` models the polarized brightness distribution of the source as a
*neural field*: a coordinate-based multi-layer perceptron (MLP), with weights
:math:`W`, that maps space, time, and frequency coordinates to the polarimetric 
quantities at that point,

.. math::

   [I, m_\ell, \chi, m_c]_W (x, y, t, f)
   = \mathrm{MLP}_W(x, y, t, f).

The network is evaluated on a grid of coordinates to produce an image, video, or 
spectral cube. A spatial Fourier transform is applied to the output of the 
network to predict interferometric data products, and the weights of the network
are updated by gradient descent to minimize the :math:`\chi^2` between predicted
and observed data products,

.. math::

   \mathcal{L} = \sum_D \chi^2_D
   = \sum_D \left( \frac{1}{N_t} \sum_{j=1}^{N_t} \frac{1}{k_D N_{D,j}}
     \sum_{i}^{N_{D,j}} \frac{|D_{ij} - \hat D_{W\,ij}|^2}{\sigma^2_{D,ij}}
     \right),

where :math:`D` runs over the chosen data products, :math:`j` over observed
times and :math:`i` over the data at each time. 

There is no explicit image prior: regularization comes from the spectral bias of 
the MLP, which favours smooth structure along all of the input coordinates.
Because the representation is continuous, the trained network can be re-sampled
at any coordinate, such as a finer pixel grid, or times at which no observation
exists.

.. note::

   The same machinery works whether the third coordinate is time (dynamic and
   multi-epoch imaging), frequency (spectral imaging), or absent altogether
   (static imaging).


Code structure
--------------

In general a ``kine`` imaging script includes the following blocks. 

.. code-block:: text

   1.  Imports and the asynchronous plotting worker
   2.  Command-line arguments, YAML hyperparameters, RNG seeds
   3.  Loading and pre-processing observation(s)
   4.  Coordinate grid
   5.  Data products
   6.  Neural field, optimizer, and training state
   7.  Initialization target and initialization training
   8.  Gain fitting variables (optional)
   9.  Training loop
   10. Saving and re-sampling

If an imaging pipelines requires multiple imaging rounds, blocks 4--10 are 
repeated once per round (see e.g. :ref:`dynamic-imaging`).


1. Imports and the plotting worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import queue
   import argparse
   import warnings
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

   warnings.filterwarnings('ignore')

   # Initialize async worker
   q = queue.Queue()
   ut.init_worker(vi.Video.async_plot, q)

The five ``kine`` modules are always imported under these short aliases. The
last two lines start a background thread that draws diagnostic figures while
training continues on the GPU: during the loop you push a dictionary of arrays
onto ``q`` and the worker renders the PNG on the CPU without blocking. The 
asynchronous worker is not necessary if saving the images takes little time or 
is not a bottleneck in the total runtime.

Use ``vi.Video.async_plot`` when the reconstruction has a third coordinate
(dynamic, multi-epoch, spectral) and ``vi.Image.async_plot`` for static imaging.


2. Arguments, hyperparameters, seeds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   par = argparse.ArgumentParser()
   par.add_argument('-obs', type=str, help='uvfits file name')
   par.add_argument('-yml', type=str, help='hyperparameter file name')
   par = par.parse_args()

   with open(par.yml, 'r') as f:
       h = yaml.safe_load(f)
   h = ut.HyperParams(h)

   rkey = jax.random.PRNGKey(h.seed)
   np.random.seed(h.seed)

``-obs`` is a path to a single UV-FITS file, or a directory when several
observations are imaged together. ``-yml`` is the parameter file.

:class:`kine.utils.HyperParams` simply turns the parsed YAML dictionary into an
object, so that ``h.npix`` can be written instead of ``h['npix']``. It performs
no validation: a key that is missing from the YAML file raises an
``AttributeError`` where it is used, and any key you add to the YAML file is
available on ``h`` without further wiring. This is what makes it easy to
parameterize a script of your own.

Seeding both JAX and NumPy makes a run reproducible: the same seed gives the
same network initialization and therefore the same reconstruction. Change the 
seed to explore the (minimal) output variability due to different realizations 
of the random initial parameters.


3. Loading and pre-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Observations are loaded as :class:`kine.obsdata.Obsdata` objects and minimal 
standard VLBI preprocessing can be applied. :class:`kine.obsdata.Obsdata` 
extends ``ehtim``'s ``Obsdata``, so every ``ehtim`` method remains available. 
The ``kine`` versions of the pre-processing methods return a ``kine`` object 
rather than an ``ehtim`` one, so they can be chained.

.. code-block:: python

   with ut.no_print():
       obs = ob.Obsdata.load_uvfits(par.obs)

       obs = obs.avg_coherent(h.tavg)           # coherent time averaging (s)
       obs = obs.add_fractional_noise(h.syserr) # systematic noise budget
       obs = obs.flag_UT_range(                 # keep/drop a UT window
           UT_start_hour=h.tflag['t0'],
           UT_stop_hour=h.tflag['t1'],
           output=h.tflag['out']
       )
       obs = obs.flag_empty()                   # drop antennas with no data
       obs = obs.norm_to_max()                  # normalize amplitudes

:func:`kine.utils.no_print` is a context manager that silences ``ehtim``'s
verbose output.

Available pre-processing
........................

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - What it does
   * - :meth:`~kine.obsdata.Obsdata.avg_coherent`
     - Coherently averages the data over ``tavg`` seconds. Reduces data volume
       (and therefore memory and runtime) and raises the SNR per point. Set
       ``tavg: 0`` to leave the data as they are.
   * - :meth:`~kine.obsdata.Obsdata.add_fractional_noise`
     - Adds a fraction of the visibility amplitude to the uncertainties, as a
       systematic noise budget. 
   * - :meth:`~kine.obsdata.Obsdata.flag_UT_range`
     - Keeps (``output='kept'``) or removes (``output='flagged'``) data in a UT
       window. Used to trim the edges of a track, or to remove a scan in which a
       key antenna is missing.
   * - :meth:`~kine.obsdata.Obsdata.flag_sites`
     - Removes named telescopes, e.g. a single-polarization station before a
       polarimetric run.
   * - :meth:`~kine.obsdata.Obsdata.flag_bl`
     - Removes a named baseline.
   * - :meth:`~kine.obsdata.Obsdata.flag_uvdist`
     - Removes data outside a range of uv-distance.
   * - :meth:`~kine.obsdata.Obsdata.flag_empty`
     - Drops antennas left in the array table with no measurements. Worth
       calling after any other flagging.
   * - :meth:`~kine.obsdata.Obsdata.norm_to_max`
     - Divides all visibilities and sigmas by the shortest-baseline flux
       density, so that fluxes are expressed in units of the peak zero-baseline
       flux. Required for the static + dynamic decomposition, which assumes
       normalized components.

For more details on data preprocessing see `eht-imaging's documentation 
<https://achael.github.io/eht-imaging/obsdata.html>`_.

Splitting into snapshots
........................

Dynamic imaging of a single observation requires the data to be grouped by time:

.. code-block:: python

   obslist = obs.split_obs(min_bl=h.min_bl)

:meth:`~kine.obsdata.Obsdata.split_obs` returns a list of ``Obsdata``, one per
snapshot, and each snapshot is matched to a video frame. ``min_bl`` drops 
snapshots formed by fewer than ``min_bl`` antennas. This parameters should be 
set to 3 when using closure phase or 4 when using closure quantities. The 
optional ``group`` argument merges each snapshot with its ``group`` neighbours 
on either side, which raises the per-frame :math:`uv`-coverage at the cost of 
temporal resolution.

Total flux
..........

Closure quantities carry no information about the total flux of a frame, so when
imaging with them the total flux must be supplied as a constraint:

.. code-block:: python

   totflux = obs.get_zbl()                       # single number
   lcurve  = obs.get_lightcurve(min_bl=h.min_bl) # one number per frame

:meth:`~kine.obsdata.Obsdata.get_zbl` returns the maximum amplitude on the
shortest baseline. :meth:`~kine.obsdata.Obsdata.get_lightcurve` splits the
observation, takes the shortest-baseline flux in each snapshot and returns a
smoothing-spline interpolation of it, optionally evaluated at times supplied
through ``times=``. When several separate observations are imaged together, the
light curve is instead built epoch by epoch:

.. code-block:: python

   lcurve = jnp.array([o.get_zbl() for o in obslist])

Imaging multiple observations
.............................

When the input is more than one observation, each file is loaded in a loop and 
the metadata are matched to a reference observation:

.. code-block:: python

   obslist = []
   for path in sorted(glob.glob(par.obs + '*.uvfits')):
       with ut.no_print():
           o = ob.Obsdata.load_uvfits(path)
           o = o.avg_coherent(h.tavg)
           o = o.add_fractional_noise(h.syserr)
           obslist.append(o)

   for i, _ in enumerate(obslist):
       obslist[i].fix_multiepoch(obslist[-1])

:meth:`~kine.obsdata.Obsdata.fix_multiepoch` copies source name, coordinates,
frequency, bandwidth, time type and polarization representation from the
reference. :meth:`~kine.obsdata.Obsdata.fix_multifreq` does the same but leaves
frequency and bandwidth untouched.


4. Coordinate grid
~~~~~~~~~~~~~~~~~~

The network is trained on an explicit grid of input coordinates built by
:func:`kine.utils.get_grid`:

.. code-block:: python

   fov = h.fov_uas * eh.RADPERUAS

   # 2D grid (x, y) -- static imaging
   grid = ut.get_grid(h.npix, h.npix)

   # 3D grid (t, x, y) -- dynamic, multi-epoch, spectral imaging
   times = ut.list_to_jaxarr([o.tstart for o in obslist])
   grid = ut.get_grid(h.npix, h.npix, len(obslist), times=times)

The 2D grid has shape ``(npix*npix, 2)`` with columns ``(x, y)``; the 3D grid
has shape ``(nt, npix*npix, 3)`` with columns ``(t, x, y)``. **The column order
matters**, because the positional encoding degrees are given per column.

Spatial coordinates are normalized to :math:`[0, 1]`. Time coordinates are
normalized to :math:`[0, 1/\mathrm{tdil}]`, where ``tdil`` (default 10) is a time
dilation factor: it sets the scale of the time axis relative to the spatial
axes, and therefore how readily the network varies its output along time
compared to space.

Passing ``times=`` places the time coordinates at the real (generally irregular)
observation times, rescaled to that interval. Passing only ``nt`` places them
regularly. Training uses the real times; re-sampling the trained network at
regular times is what produces a temporally uniform output video.

``fov`` is the field of view in radians (``eh.RADPERUAS`` converts from
microarcseconds) and ``npix`` the number of pixels per side, so the pixel size is
``fov/npix``. Both enter the reconstruction only through the grid and through
the image metadata created in the next block.

Finally, the number of output channels is set from the requested data products:

.. code-block:: python

   outdim = 1
   if 'visQ' in h.data_prod: outdim = 4
   if 'visV' in h.data_prod: outdim = 5


5. Data products
~~~~~~~~~~~~~~~~

``kine`` never fits an image to an image: it fits the observed interferometric
data products. Which ones to use is set by ``data_prod`` in the YAML file, as a
list of string codes. Each code is a product name followed by a single letter
naming the Stokes parameter.

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Code
     - Quantity
     - Notes
   * - ``visI``, ``visQ``, ``visU``, ``visV``
     - Complex visibilities :math:`V_{AB}`
     - Carry the full information content. Correct choice when the data are
       well calibrated (or have been self-calibrated).
   * - ``ampI``
     - Visibility amplitudes :math:`|V_{AB}|`
     - Immune to phase errors, sensitive to amplitude gains.
   * - ``logampI``
     - :math:`\log|V_{AB}|`
     - As above, with better-behaved gradients over a wide dynamic range.
   * - ``cphaseI``
     - Closure phases :math:`\arg(V_{AB}V_{BC}V_{CA})`
     - Invariant under station-based phase errors. Carry no information on the
       absolute source position, so a reconstruction may drift within the frame.
   * - ``logcampI``
     - Log closure amplitudes
     - Invariant under station-based amplitude errors. Carry no information on
       the total flux, hence the light-curve constraint.
   * - ``bsI``
     - Bispectra :math:`V_{AB}V_{BC}V_{CA}`
     - Alternative to closure phases, retaining amplitude information.
   * - ``mbreve``
     - :math:`\breve m = (\tilde Q + i\tilde U)/\tilde I`
     - Complex polarization ratio, for polarimetric imaging.

The loss terms from all listed data products are summed with equal weight.

**Direct Fourier transform**

.. code-block:: python

   improxy = eh.image.make_square(obs, h.npix, fov, pol_prim='I')

   data = {}
   for dtype in h.data_prod:
       target, sigma, A, padmask = ob.Obsdata.get_data(obslist, dtype, improxy)
       data[dtype] = {
           'target': target,
           'sigma': sigma,
           'A': A,
           'padmask': padmask
       }

``improxy`` is an empty ``ehtim`` image that carries only metadata (pixel size,
field of view, source coordinates); it tells ``ehtim`` how to build the Fourier
operators, and is never itself reconstructed.

:meth:`~kine.obsdata.Obsdata.get_data` returns ``(target, sigma, A)`` for a
single ``Obsdata`` and ``(target, sigma, A, padmask)`` for a list of snapshots.
``A`` holds the DFT matrices that map image pixels to visibilities — one matrix
for a direct product, three for closure phases and bispectra, four for closure
amplitudes. Because snapshots contain different numbers of visibilities, the
arrays are padded to a common length and ``padmask`` marks the real entries, so
that padding contributes nothing to the loss.

**NUFFT**

.. code-block:: python

   data = {}
   for dtype in h.data_prod:
       target, sigma, padmask = ob.Obsdata.get_data_nfft(obslist, dtype, improxy)
       data[dtype] = {'target': target, 'sigma': sigma, 'padmask': padmask}

   def prepare_nufft(o):
       bl = o.get_baselines_nfft()
       uv = o.get_uvpoints(improxy.psize)
       uvind = o.get_uvpoints(improxy.psize, conj=False)
       pulse = o.get_pulsefac(uv, ehc.PULSE_DEFAULT)
       tria = o.get_closure_indices(bl, which='triangles')
       quad = o.get_closure_indices(bl, which='quadrangles')
       return uv['u'], uv['v'], uvind['u'], pulse, tria, quad

   with ThreadPoolExecutor() as ex:
       u, v, uvind, pulse, tria, quad = map(
           list, zip(*ex.map(prepare_nufft, obslist))
       )

   uv = {'u': ut.pad(u).astype('float32'),
         'v': ut.pad(v).astype('float32')}
   uvind = ut.map_val_to_ind(uv['u'], ut.pad(uvind).astype('float32'))
   pulse = ut.pad(pulse).astype('complex64')
   tria  = ut.pad(tria).astype('int32')
   quad  = ut.pad(quad).astype('int32')

Here no DFT matrix is built. Instead the scaled :math:`uv` points, the pulse
factors, and the index arrays that assemble baselines into triangles and
quadrangles are precomputed, and the visibilities are evaluated at run time with
a non-uniform FFT (``jax-finufft``). ``ut.pad`` pads the per-epoch lists into
rectangular JAX arrays; the ``ThreadPoolExecutor`` is only there to speed up
this (purely CPU-side) preparation.

.. admonition:: Which one to use
   :class: tip

   **Prefer the direct DFT.** Switch to the NUFFT only when memory becomes a
   problem, which happens when there are many data points and many epochs: the
   DFT stores a dense ``(nvis × npix²)`` complex matrix per snapshot and per
   Fourier operator, so its footprint grows quickly with the number of pixels,
   visibilities and frames.

   Note that the NUFFT path currently reconstructs Stokes I only, does not
   support bispectra or ``mbreve``, and does not support simultaneous gain
   fitting.


6. The neural field, optimizer, and training state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   network = mo.NeuralField(
       posenc_deg=tuple(h.nposenc),  # per input coordinate
       outdim=outdim,                # 1, 4 or 5 output channels
       depth=h.depth,                # hidden layers
       width=h.width,                # neurons per hidden layer
       activ=nn.gelu,                # hidden activation
       outactiv=nn.softplus,         # output activation for Stokes I
       outshift=h.outshift,          # shift before the output activation
       scaling_i=h.scaling_i         # Stokes I output scaling
   )
   params      = network.init(rkey, jnp.ones([grid.shape[-1]]), train=True)
   batch_stats = network.init(rkey, jnp.ones([grid.shape[-1]]), train=True)

:class:`kine.model.NeuralField` is an MLP with batch normalization on every
hidden layer and a residual skip connection from layer ``skipat`` (default: the
first) to the output layer.

**Positional encoding.** Before the first layer, the input coordinates are
concatenated with a Fourier feature expansion,

.. math::

   x \rightarrow \left[x, \sin(x), \cos(x), \sin(2x), \cos(2x), \ldots,
   \sin(2^{\mathrm{deg}}x), \cos(2^{\mathrm{deg}}x)\right],

with a separate degree per input coordinate, given by ``posenc_deg``. A degree
of 0 leaves that coordinate unencoded, so the network sees it raw and its
spectral bias suppresses fast variation along it. Higher degrees make it easier
for the network to represent rapid variation along that coordinate.

The tuple must have one entry per grid column: ``(t, x, y)`` for a 3D grid,
``(x, y)`` for a 2D grid. When a 2D static network is built alongside a 3D one,
the script slices the same YAML entry: ``posenc_deg=tuple(h.nposenc[-2:])``.

**Output channels.** ``outdim`` selects which polarimetric quantities the
network predicts:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - ``outdim``
     - Output channels
   * - ``1``
     - :math:`\hat I` only
   * - ``4``
     - :math:`\hat I`, :math:`\hat m_\ell`, :math:`\sin 2\hat\chi`,
       :math:`\cos 2\hat\chi`
   * - ``5``
     - as above plus :math:`\hat m_c`

The EVPA is predicted through its sine and cosine to avoid a discontinuity at
the wrap, and recovered as
:math:`\hat\chi = \tfrac12 \arctan2(\sin 2\hat\chi, \cos 2\hat\chi)`. The Stokes
parameters follow as

.. math::

   \hat Q = -\hat I\,\hat m_\ell \sin 2\hat\chi, \qquad
   \hat U =  \hat I\,\hat m_\ell \cos 2\hat\chi, \qquad
   \hat V =  \hat I\,\hat m_c .

**Activations.** Hidden layers use ``activ``, GELU by default.
:func:`kine.model.sharpgelu` is a sharper variant, used through
``partial(mo.sharpgelu, s=3)``, that favours more compact structure. The Stokes I
channel is passed through ``outactiv`` — ``nn.softplus`` for a single network,
``nn.sigmoid`` when the output is a normalized component of a decomposition —
after subtracting ``outshift``, which starts the network near zero brightness so
that emission has to be built up by the data rather than removed from an
initially bright frame. ``scaling_i`` multiplies the Stokes I channel;
``scaling_ml`` caps the linear polarization fraction. The remaining channels use
fixed sigmoids and are not configurable.

**Optimizer and training state.**

.. code-block:: python

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

:class:`kine.trainer.Trainer` is Flax's ``TrainState`` extended with batch
normalization statistics. The optimizer is ``Adamax``. The schedule above keeps
a large learning rate through the initialization phase and drops it by the given
factor once data-driven training starts; :class:`kine.utils.Schedule` provides a
smooth exponential alternative, used through ``sched.exponential``.


7. Initialization
~~~~~~~~~~~~~~~~~

Before fitting any data, the network is trained to reproduce a simple image.
This is a plain pixel-to-pixel regression,

.. math::

   \mathcal{L}_\mathrm{init} = \sum_{i,j}
   \left(I_\mathrm{init}(x_i, y_i) - \hat I_W(x_i, y_i, t_j)\right)^2,

with no Fourier transform involved. When imaging with closure phases — which do
not constrain absolute position — it anchors the source in the frame; more
generally it avoids starting the fit from a poor local minimum. What matters is
that the initialization has roughly the right total flux and covers the region
where the emission is expected; its detailed shape does not affect the converged
result.

The target is built with a :class:`kine.video.Video` (or
:class:`kine.video.Image`) object:

.. code-block:: python

   init_vid = vi.Video(times, h.npix, fov, obs.ra, obs.dec, h.initniter)
   init_vid.add_tophat(lcurve, h.init_params)
   init_vid.plot()

:meth:`~kine.video.Video.add_tophat` builds a blurred disk carrying the
light-curve flux in each frame, from ``init_params``: ``fwhm`` (disk diameter in
µas), ``blur`` (Gaussian blurring in µas), and ``posx``/``posy`` (offsets in
pixels). Alternatives are :meth:`~kine.video.Video.from_h5` (start from a
previous reconstruction), :meth:`~kine.video.Video.add_video_i` (load a fixed
Stokes I video) and :meth:`~kine.video.Video.add_constant_linpol` /
:meth:`~kine.video.Video.add_constant_circpol`.

The initialization loop then looks exactly like a training loop, except that the
target is an array rather than a set of data products:

.. code-block:: python

   init = vi.Video(times, h.npix, fov, obs.ra, obs.dec, h.initniter)
   lloss, loss = [], 0

   for i in (pbar := tqdm(range(1, h.initniter+1))):
       pbar.set_description(f'Loss {loss:.1e}')

       loss, _, out, state = tr.Trainer.train_step(
           odict(
               state=state,
               grid=grid,
               init_arr=init_vid.iarr
           )
       )
       lloss.append(loss)

       if i == 1 or i % 500 == 0 or i == h.initniter:
           q.put(dict(video=init, out=out, loss=lloss,
                      outpath='./output_init.png'))

   q.join()

``q.put`` hands a snapshot to the plotting thread; ``q.join()`` waits for the
queue to drain before the script moves on.


8. Gain fitting
~~~~~~~~~~~~~~~

Station gains can be fitted jointly with the image, as learnable parameters:

.. code-block:: python

   sites, nsites, nvis, bl_indx, lower, upper = obs.set_gains_vars(
       obslist, h.gains_prior
   )

   ag_schedule = ut.Schedule(5e-5, 1e-3, h.niter)
   ag_network = mo.AmplitudeGains(
       nsites=nsites, ntimes=ntimes, lower=lower, upper=upper
   )
   ag_params = ag_network.init(
       rkey, jnp.ones((ntimes, nvis, 2), dtype=int), jnp.ones((ntimes), dtype=int)
   )
   ag_state = tr.Trainer.create(
       apply_fn=ag_network.apply,
       params=ag_params['params'].unfreeze(),
       tx=optax.adamax(learning_rate=ag_schedule.exponential)
   )

   pg_network = mo.PhaseGains(nsites=nsites, ntimes=ntimes)
   pg_params = pg_network.init(
       rkey, jnp.ones((ntimes, nvis, 2), dtype=int), jnp.ones((ntimes), dtype=int)
   )
   pg_state = tr.Trainer.create(
       apply_fn=pg_network.apply,
       params=pg_params['params'].unfreeze(),
       tx=optax.adamax(learning_rate=1)
   )

:meth:`~kine.obsdata.Obsdata.set_gains_vars` reads ``gains_prior`` — a
per-telescope ``[lower, upper]`` range of allowed multiplicative amplitude
corrections — and returns the bookkeeping needed to map each visibility to the
pair of stations that formed it. :class:`kine.model.AmplitudeGains` holds one
gain per station and per frame, initialized to 1 and clipped to its allowed
range at every step; :class:`kine.model.PhaseGains` holds one phase per station
and per frame, initialized to 0 and wrapped to :math:`[-\pi, \pi]`.

Gains are applied to the *data*, not to the model. Amplitude gains are applied
when ``ampI`` or ``logampI`` is among the data products, and amplitude and phase
gains together when ``visI`` is. Closure quantities are gain-invariant by
construction and are left untouched, so a run using only closure products
carries the gain networks along without them affecting the fit.

.. note::

   Simultaneous gain fitting is currently wired into the static + dynamic
   decomposition path only (block 9, the ``s_grid`` branches). The single-network
   and NUFFT paths ignore gain states.


9. The training loop
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   tr.NPIX = h.npix

   video = vi.Video(times, h.npix, fov, obs.ra, obs.dec, h.niter)
   lloss, loss = {dp: [] for dp in h.data_prod} | {'lcurve': []}, 0

   for i in (pbar := tqdm(range(1, h.niter+1))):
       pbar.set_description(f'Loss {loss:.1e}')

       loss, ldict, out, state = tr.Trainer.train_step(
           odict(
               state=state,
               grid=grid,
               data=data,
               lcurve=lcurve
           )
       )

       for l in lloss:
           lloss[l].append(ldict[l])

       if i == 1 or i % 500 == 0 or i == h.niter:
           q.put(dict(video=video, out=out, loss=lloss,
                      outpath='./output_train.png'))

   q.join()

:meth:`kine.trainer.Trainer.train_step` is a single ``@jax.jit``-compiled
function that evaluates the network on the grid, computes the loss, takes the
gradient with respect to every training state passed in, applies the update, and
refreshes the batch-norm statistics.

.. important::

   ``tr.NPIX`` is a module-level global that must be set to the current ``npix``
   before training. It is needed by the NUFFT and by the border regularizer, and
   it must be updated at every step of a multi-resolution pipeline.

**Why an OrderedDict.** ``train_step`` takes a single ``OrderedDict`` because
``jax.jit`` does not preserve the order of keyword arguments. Everything the
loss needs is passed through it, and its contents determine both the loss
function used and the shape of the return value. Any key containing ``state`` is
treated as a training state and receives gradients, **in the order in which the
keys appear**, which is also the order in which the updated states are returned.

**Which loss is used.** The dispatch is by presence of keys, tested in this
order:

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Key present
     - Loss
     - Purpose
   * - ``init_arr``
     - initialization
     - Regress the network onto an image or video array (2D or 3D).
   * - ``init_vid_ml``
     - polarimetric initialization
     - Regress the polarization channels only, with Stokes I ignored.
   * - ``uvpoints``
     - NUFFT training
     - Stokes I data products evaluated with a non-uniform FFT.
   * - ``init_vid_i``
     - polarimetric training
     - Fit Q and U with Stokes I held fixed at a given video.
   * - ``grid``
     - single-network training
     - Static or dynamic imaging with one neural field, full polarization.
   * - ``s_grid`` + ``min_lcurve``
     - decomposition, fluxes assigned
     - Static + dynamic decomposition with a known static flux, plus gains.
   * - ``s_grid`` alone
     - decomposition, flux regularized
     - As above, but the static flux is found through regularization.

.. warning::

   ``grid`` is tested before ``s_grid``. A decomposition run must therefore pass
   ``s_grid`` and ``d_grid`` and **not** ``grid``, or it will silently fall back
   to the single-network loss.

**Return value.** ``train_step`` returns ``loss, ldict, *outputs, *states``. For
every loss except the decomposition ones there is a single output array and a
single state:

.. code-block:: python

   loss, ldict, out, state = tr.Trainer.train_step(...)

while the decomposition losses return the two components and their sum, followed
by the four updated states in the order they were passed:

.. code-block:: python

   loss, ldict, s_out, d_out, out, s_state, d_state, ag_state, pg_state = \
       tr.Trainer.train_step(...)

``ldict`` maps each loss term to its current value, and is ``None`` during
initialization. Its keys are the data product codes plus the active
regularizers, which is why ``lloss`` is built as
``{dp: [] for dp in h.data_prod} | {...}``: the extra keys must match the
regularizers of the branch in use.

**Regularizers.**

.. list-table::
   :header-rows: 1
   :widths: 18 26 56

   * - Key
     - Where
     - What it does
   * - ``lcurve``
     - single-network, NUFFT
     - Constrains the total flux of each frame to the light curve. Needed
       whenever closure amplitudes are used, since they carry no flux
       information.
   * - ``border``
     - decomposition
     - Penalizes flux in the outer ``npix/20`` rows and columns, keeping the
       source away from the frame edge. Weight ``w_border`` (default ``1e3``).
   * - ``s_flux``, ``d_flux``
     - decomposition, fluxes assigned
     - Force the static image and each dynamic frame to sum to unity, so that
       the physical flux is carried entirely by the light curve. Weight
       ``w_flux`` (default ``1e3``).
   * - ``min_dyn``
     - decomposition, flux regularized
     - Minimizes the persistent flux of the dynamic component, pushing
       time-constant emission into the static network. Weight ``w_flux``
       (default ``5``).
   * - ``overlap``
     - polarimetric
     - Suppresses polarized emission where there is no total intensity.

Weights are overridden by passing them into the dict, e.g. ``w_border=0``.


10. Saving and re-sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the trained network is continuous, the final result is usually not the
array produced by the last training iteration but a fresh evaluation on a finer
grid:

.. code-block:: python

   # Quick look at the last training output
   video.from_video(out, loss=lloss)
   video.plot_gif(outpath='./output.gif')

   # Re-sample the network on a finer grid
   grid_out = ut.get_grid(h.npix_out, h.npix_out, ntimes, times=times)
   video = vi.Video(times, h.npix_out, fov, obs.ra, obs.dec, h.niter)
   video.from_state(state, grid_out)
   video.save_h5('./output.h5')

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Method
     - Use
   * - :meth:`~kine.video.Video.from_video` / :meth:`~kine.video.Image.from_image`
     - Populate the object from an array returned by ``train_step``.
   * - :meth:`~kine.video.Video.from_state`
     - Evaluate a trained network on an arbitrary grid.
   * - :meth:`~kine.video.Video.from_states`
     - Evaluate and recombine the static and dynamic networks.
   * - :meth:`~kine.video.Video.plot`
     - Diagnostic figure: frames, loss curves, and polarization panels when
       present. ``scale='log'`` with ``drange`` for high dynamic range.
   * - :meth:`~kine.video.Video.plot_gif`
     - Animated GIF of the video.
   * - :meth:`~kine.video.Video.save_h5`
     - HDF5 video, readable by ``ehtim``.
   * - :meth:`~kine.video.Video.save_fits` / :meth:`~kine.video.Image.save_fits`
     - FITS image.
   * - :meth:`~kine.video.Video.save_gains`
     - Text file of fitted amplitude gains.

Re-sampling on a finer grid is genuinely cheap — it is one forward pass — so the
training resolution can be kept modest while the output is written at higher
resolution. The same applies to time: passing a regular ``times`` array to
``get_grid`` produces evenly spaced frames from irregularly sampled
observations.


Imaging scenarios
-----------------

The following sections describe how the ten blocks change for each imaging
scenario. Each corresponds to a script in ``scripts/`` and a parameter file in
``parameters/``.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Scenario
     - Script
     - Third coordinate
     - Networks
   * - :ref:`static-imaging`
     - ``example_static_imaging.py``
     - none
     - 1 (2D)
   * - :ref:`spectral-imaging`
     - ``example_spectral_imaging.py``
     - frequency
     - 1 (3D)
   * - :ref:`multiepoch-imaging`
     - ``example_multiepoch_imaging.py``
     - time (days--years)
     - 1 (3D)
   * - :ref:`dynamic-imaging`
     - ``example_dynamic_imaging.py``
     - time (within a track)
     - 2 (2D + 3D) + gains
   * - :ref:`polarimetric-imaging`
     - ``example_dynamic_imaging_pol.py``
     - time (within a track)
     - 1 (3D, polarization only)


.. _static-imaging:

Static imaging
~~~~~~~~~~~~~~

Reconstruct a single image from a single observation. This is the simplest
scenario and the best starting point.

**What changes**

- *Block 1* — use ``vi.Image.async_plot`` and the :class:`kine.video.Image`
  class throughout, instead of :class:`kine.video.Video`.
- *Block 3* — no ``split_obs``: the whole observation is fitted as one block.
  The flux constraint is the single number ``totflux = obs.get_zbl()``.
- *Block 4* — a 2D grid, ``ut.get_grid(h.npix, h.npix)``. ``nposenc`` therefore
  has **two** entries, one for ``x`` and one for ``y``
  (``params_static_imaging.yml`` uses ``[0, 0]``).
- *Block 5* — ``get_data`` is called with a single ``Obsdata``, so it returns
  ``(target, sigma, A)`` and there is no ``padmask``. The 2D loss functions are
  selected automatically by the absence of that key.
- *Block 7* — ``init_im.add_tophat(totflux, h.init_params)`` takes a scalar flux
  rather than a light curve.
- *Block 9* — the light curve is a one-element array:

  .. code-block:: python

     loss, ldict, out, state = tr.Trainer.train_step(
         odict(
             state=state,
             grid=grid,
             data=data,
             lcurve=jnp.array([totflux])
         )
     )

- *Block 10* — re-sample on a finer 2D grid and write FITS:

  .. code-block:: python

     grid_out = ut.get_grid(h.npix_out, h.npix_out)
     image = vi.Image(h.npix_out, fov, obs.ra, obs.dec, h.niter)
     image.from_state(state, grid_out)
     image.save_fits('./output_image.fits')

**Options**

``data_prod`` is the main choice. With well-calibrated data, ``[visI]`` uses all
the available information. With residual gains, use
``[cphaseI, logcampI]`` — the closure pair used in
``params_static_imaging.yml`` — and rely on the ``lcurve`` term to fix the total
flux, and on the initialization to fix the position in the frame. Adding
``visQ``/``visU`` to the list raises ``outdim`` to 4 and reconstructs linear
polarization simultaneously; adding ``visV`` raises it to 5.


.. _spectral-imaging:

Spectral imaging
~~~~~~~~~~~~~~~~

Reconstruct the frequency dependence of the source from several observations at
different frequencies. Frequency plays exactly the role that time plays in
dynamic imaging: the network learns a smooth interpolation across the frequency
axis, so the reconstruction can be sampled at frequencies that were not
observed.

**What changes**

- *Block 3* — load one file per frequency into ``obslist`` and match the
  metadata with :meth:`~kine.obsdata.Obsdata.fix_multifreq`, which leaves each
  observation's own ``rf`` and bandwidth intact. The flux constraint is built
  per band, ``lcurve = jnp.array([o.get_zbl() for o in obslist])``.
- *Block 4* — the third coordinate is the observing frequency:

  .. code-block:: python

     freqs  = jnp.array([o.rf for o in obslist])
     labels = ['B1 (214 GHz)', 'B2 (216 GHz)',
               'B3 (226 GHz)', 'B4 (228 GHz)']
     grid = ut.get_grid(h.npix, h.npix, len(freqs), times=freqs)

  ``labels`` is passed to the ``Video`` constructor as ``dates=labels`` and is
  used to caption the frames; without it the frames are labelled by the raw
  coordinate value.
- *Block 5* — the NUFFT path, since spectral runs are usually at high
  resolution (``params_multifreq_imaging.yml`` uses ``npix: 100``,
  ``npix_out: 200``).
- *Block 6* — ``outdim = 1``: spectral imaging is currently Stokes I only, which
  is also what the NUFFT loss supports. ``nposenc: [0, 0, 0]`` leaves the
  frequency axis unencoded, so the spectral behaviour stays smooth.
- *Block 9* — pass the NUFFT variables:

  .. code-block:: python

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

**Options**

The output is a cube of images, one per observed frequency, and the network can
be re-sampled at intermediate frequencies. There is no explicit spectral-index
model: the spectral dependence is whatever the neural field interpolates between
the observed bands. Explicit spectral modelling, and combining the spectral and
multi-epoch axes, are planned developments.


.. _multiepoch-imaging:

Multi-epoch imaging
~~~~~~~~~~~~~~~~~~~

Reconstruct the evolution of a source across many separate observations,
spanning days to decades — for example a monitoring programme such as MOJAVE.
Each epoch contributes one frame, and imaging all epochs together means each
frame is constrained by far more data than it would be alone, which is where the
resolution and dynamic range gains come from.

**What changes**

- *Block 3* — glob the input directory, load each file, match metadata with
  :meth:`~kine.obsdata.Obsdata.fix_multiepoch`, and guard against empty scan
  tables:

  .. code-block:: python

     for i, _ in enumerate(obslist):
         obslist[i].fix_multiepoch(obslist[-1])
         if obslist[i].scans is not None and len(obslist[i].scans) == 0:
             obslist[i].scans = None

  Wrapping the load in ``try``/``except`` is worthwhile: with hundreds of
  archival files, a few are usually unreadable, and the script reports them
  rather than aborting.
- *Block 4* — times come from the file metadata rather than from ``tstart``:

  .. code-block:: python

     times = ut.get_times_multiepoch(obspath)
     dates = ut.get_times_multiepoch(obspath, labels=True)
     grid  = ut.get_grid(h.npix_1, h.npix_1, len(obslist), times=times)

  :func:`kine.utils.get_times_multiepoch` reads the MJD from each UV-FITS
  header, so no assumption is made about file naming; a
  ``datetime.strptime`` pattern can be supplied through ``fmt=`` to parse the
  dates from the file names instead, and a list of already-loaded ``Obsdata``
  can be passed instead of paths. ``labels=True`` returns ``YYYY-MM-DD`` strings
  for plot captions; the default rounds down to integer MJD, one coordinate per
  day. Pass ``dates=dates`` to the ``Video`` constructor.
- *Block 5* — the NUFFT path. This is the scenario the NUFFT exists for:
  ``params_multiepoch_imaging.yml`` uses ``npix: 300`` over a 1000 µas field
  with of order a hundred epochs, where the DFT matrices would not fit in
  memory.
- *Block 7* — the initialization disk is typically offset (``posx: -50``,
  ``posy: 50``) to place a one-sided jet sensibly within the frame, since
  closure phases do not constrain absolute position.
- *Block 9* — as for spectral imaging, with the NUFFT variables in the dict.
  Multi-epoch runs are long (``niter: 30000``); the diagnostic plots use
  ``scale='log'`` with a ``drange`` to show the faint extended emission.

**Options**

``nposenc: [4, 0, 0]`` encodes the time axis so that the network can follow
structural change between epochs, while leaving the spatial axes to the MLP's
own spectral bias. Since the sampling is irregular, whether ``get_grid`` receives
the true MJDs matters: the network learns as a function of real elapsed time, so
interpolated frames are correctly placed. In this scenario ``tavg: 0`` and
``min_bl: 0`` are usual — each epoch is a full track and there is nothing to
discard.


.. _dynamic-imaging:

Dynamic imaging
~~~~~~~~~~~~~~~

Reconstruct a video from a *single* observation of a source that varies within
the track, such as Sgr A* with the EHT. Here the instantaneous coverage is far
too sparse to constrain individual frames, so the reconstruction relies entirely
on sharing information across time.

This scenario uses two extensions of the basic structure: a decomposition into a
persistent and a variable component, and a three-step pipeline.

**Static + dynamic decomposition**

The source is modelled as a persistent image plus a time-variable video,

.. math::

   I(x, y, t) = S_\mathrm{static}\, f_\mathrm{static}(x, y)
              + \big(L(t) - S_\mathrm{static}\big)\, f_\mathrm{dynamic}(x, y, t),

where :math:`f_\mathrm{static}` is a 2D neural field, :math:`f_\mathrm{dynamic}`
a 3D one, :math:`L(t)` is the light curve, and :math:`S_\mathrm{static}` is the
flux of the persistent component. Both fields are normalized to unit total flux
by the ``s_flux`` and ``d_flux`` regularizers, so all the physical flux is
carried by the light curve. This is also why the data must be normalized with
:meth:`~kine.obsdata.Obsdata.norm_to_max` in block 3.

Both networks use ``outactiv=nn.sigmoid`` rather than softplus, since their
outputs are normalized fractions; the static network is shallower
(``s_depth: 4``) than the dynamic one (``d_depth: 6``), and takes the last two
entries of ``nposenc`` because its grid is 2D.

**The three-step pipeline**

The script runs blocks 4--10 three times, at increasing resolution:

.. list-table::
   :header-rows: 1
   :widths: 12 22 66

   * - Step
     - Resolution
     - Purpose
   * - 0
     - ``npix_0: 16``, 160 µas
     - Find :math:`S_\mathrm{static}`.
   * - 1
     - ``npix_1: 32``, 160 µas
     - Reconstruct the video from a disk initialization, with the flux split
       assigned.
   * - 2
     - ``npix_2: 64``, 200 µas
     - Refine, initialized from the step 1 video.

*Step 0* trains both networks with the flux-regularized loss — ``s_grid`` and
``d_grid`` in the dict, but **no** ``min_lcurve`` or ``lcurve``. The
``min_dyn`` regularizer minimizes the persistent flux in the dynamic component,
so time-constant emission accumulates in the static network. The static flux
follows from its total:

.. code-block:: python

   min_lcurve = ut.get_static_flux(s_out.sum(), lcurve.min())

:func:`kine.utils.get_static_flux` keeps the value found by the regularizer
unless it exceeds the light-curve minimum, in which case it falls back to that
minimum less a small offset, guaranteeing that
:math:`L(t) - S_\mathrm{static}` stays positive. The gain learning rate is held
at ``1e-12`` in this step (``ut.Schedule(1e-12, 1e-12, h.niter_0)``), i.e. gains
are effectively frozen while the flux split is being determined.

*Step 1* rebuilds the grids and data products at ``npix_1``, re-initializes both
networks from a disk, releases the gains (``ut.Schedule(5e-5, 1e-3, h.niter_1)``)
and trains with the assigned split:

.. code-block:: python

   loss, ldict, s_out, d_out, out, s_state, d_state, ag_state, pg_state = \
       tr.Trainer.train_step(
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

The result is written with :meth:`~kine.video.Video.from_states`, which
evaluates both networks and recombines them with the light curve, plus
``save_h5`` and ``save_gains``.

*Step 2* repeats at ``npix_2`` with three changes: the networks are initialized
from ``video_1.h5`` via :meth:`~kine.video.Video.from_h5` rather than from a
disk (``blur=0`` with ``fn=np.median`` for the static field, ``blur=30`` for the
dynamic one), the dynamic network switches to the sharper activation
(``d_network.activ = partial(mo.sharpgelu, s=3)``), and the border regularizer
is switched off with ``w_border=0``, since the source is already well centred.

.. note::

   Each step re-creates the grids, the data products, the ``improxy``, the
   network parameters and the training states, because they all depend on
   ``npix`` and ``fov``. Only the *network definitions*, the light curve and
   ``min_lcurve`` carry over. Remember to update ``tr.NPIX`` at each step.

**Options**

``gains_prior`` should reflect what is known about each station's calibration:
tight bounds for well-calibrated antennas, loose ones for stations with known
problems. ``nposenc: [6, 0, 0]`` gives the time axis a high encoding degree,
which is what allows intra-track variability to be represented. The
``initniter``/``niter`` pairs are set per step, with step 2 typically using more
initialization iterations (to reproduce a detailed video rather than a disk) and
fewer training ones.

If the source has no persistent component worth separating, the decomposition
can be dropped entirely: use a single 3D network and the single-network loss
(``grid``, ``data``, ``lcurve``), exactly as in multi-epoch imaging but with
``times`` taken from the snapshots. Note that gain fitting is not available on
that path.


.. _polarimetric-imaging:

Dynamic polarimetric imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reconstruct the linear polarization of a variable source, with Stokes I held
fixed at a previously reconstructed video. Splitting the problem this way is
what makes polarimetric reconstruction of gain-corrupted data tractable: total
intensity is reconstructed first from closure quantities, the data are
self-calibrated to that reconstruction, and the polarization is then fitted with
complex visibilities.

**What changes**

- *Block 3* — no ``norm_to_max`` (there is no decomposition to normalize), and
  stations that observed in a single polarization are flagged out, e.g.
  ``obs = obs.flag_sites(['JC'])``.
- *Block 5* — ``data_prod: [visQ, visU]``. Stokes I is not fitted, so it does
  not appear.
- *Block 6* — a different network class:

  .. code-block:: python

     sharpgelu = partial(mo.sharpgelu, s=3)

     network = mo.NeuralFieldPol(
         posenc_deg=tuple(h.nposenc),
         outdim=3,                  # (ml, sin 2xi, cos 2xi)
         depth=h.depth,
         width=h.width,
         activ=sharpgelu,
         outactiv=nn.sigmoid,
         outshift=h.outshift,
         scaling_ml=h.scaling_ml
     )

  :class:`kine.model.NeuralFieldPol` outputs only the three polarization
  channels. ``scaling_ml`` caps the linear polarization fraction
  (``0.75`` in ``params_dynamic_imaging_pol.yml``).
- *Block 7* — the initialization loads the fixed Stokes I video and adds a
  constant polarization to it, then regresses only the polarization channels:

  .. code-block:: python

     init_vid = vi.Video(times, h.npix, fov, obs.ra, obs.dec, h.initniter)
     init_vid.add_video_i(par.obs.replace('uvfits', 'hdf5'))
     init_vid.add_constant_linpol()

     loss, _, out, state = tr.Trainer.train_step(
         odict(
             state=state,
             grid=grid,
             init_vid_ml=init_vid.larr,
             init_vid_x=init_vid.xarr
         )
     )

  Note that this uses ``init_vid_ml``/``init_vid_x``, not ``init_arr``.
- *Block 9* — the fixed Stokes I video is passed as ``init_vid_i``, which is
  what selects the polarimetric loss:

  .. code-block:: python

     loss, ldict, out, state = tr.Trainer.train_step(
         odict(
             state=state,
             grid=grid,
             data=data,
             init_vid_i=init_vid.iarr
         )
     )

  Q and U are formed from the fixed I and the predicted channels,

  .. math::

     \hat Q = -\hat I\,\hat m_\ell \sin 2\hat\chi, \qquad
     \hat U =  \hat I\,\hat m_\ell \cos 2\hat\chi,

  and the ``overlap`` term, :math:`\langle |m_\ell|\,e^{-|I|/\tau}\rangle` with
  :math:`\tau = 0.01`, suppresses polarization in regions with no total
  intensity. The loss dictionary keys are therefore the data products plus
  ``overlap``.
- *Block 10* — since the network does not predict Stokes I, the total intensity
  must be attached to the output object by hand before plotting or saving:

  .. code-block:: python

     video = vi.Video(times, h.npix, fov, obs.ra, obs.dec, h.niter)
     video.iarr = init_vid.iarr.copy()
     ...
     video.from_video(out, loss=lloss)
     video.plot_gif(outpath='./out_pol.gif')
     video.save_h5('./video_pol.h5')

**Options**

``nposenc: [4, 0, 0]`` is lower than for the Stokes I run, since polarization
structure is generally smoother in time and the Q/U data are noisier. The sharp
GELU activation is used from the start rather than only in a refinement step.
Full-Stokes reconstruction in a single network is also possible — set
``outdim=5`` on a :class:`kine.model.NeuralField` and include ``visQ``,
``visU``, ``visV`` in ``data_prod`` — but it requires data whose gains have
already been solved for.


Practical notes
~~~~~~~~~~~~~~~

**Compilation.** ``train_step`` is JIT-compiled, so the first iteration of each
loop is slow while XLA compiles it. Changing the shape of anything in the
dictionary, or the set of keys, triggers a recompilation — which is why each
step of a multi-step pipeline pays that cost again.

**Memory.** The dominant cost of the direct DFT path is the Fourier operators,
which scale as ``nvis × npix²`` per snapshot and per operator (one for direct
products, three for closure phases, four for closure amplitudes). Reducing
``npix``, raising ``tavg``, or raising ``min_bl`` all reduce it; when that is not
enough, move to the NUFFT path.

**Diagnosing a run.** The per-term loss curves in the diagnostic PNG are the
main diagnostic: the :math:`\chi^2` of each data product should approach unity.
A data term stuck well above 1 points to an over-constrained model (too few
pixels, too small a field of view, too tight a gain prior); a term far below 1
points to overestimated uncertainties. A reconstruction that drifts across the
frame between epochs is expected when imaging with closure phases only and is
corrected by re-aligning the frames afterwards.

**Choosing the field of view and resolution.** The field of view must contain
all the emission, since flux outside it cannot be represented and will corrupt
the fit; the border regularizer helps keep the source inside it. The pixel size
should oversample the expected resolution, which for a forward-modelling method
like ``kine`` is finer than the nominal beam. Training at a modest resolution and
re-sampling the trained network for the output keeps runtime down without
sacrificing the resolution of the final product.
