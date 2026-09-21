===================
Parameter Reference
===================

The ``kine`` imaging scripts read their settings from a YAML configuration file,
passed on the command line with ``-yml``:

.. code-block:: bash

   python example_static_imaging.py -obs observations.uvfits -yml params.yml

The file is loaded with ``yaml.safe_load`` and wrapped in
:class:`kine.utils.HyperParams`, which exposes every top-level key as an
attribute, so that ``h.npix`` can be written instead of ``h['npix']``:

.. code-block:: python

   with open(par.yml, 'r') as f:
       h = yaml.safe_load(f)
   h = ut.HyperParams(h)

:class:`~kine.utils.HyperParams` performs **no validation and provides no
defaults**. Every key a script reads must be present in the YAML file, or an
``AttributeError`` is raised at the line where it is used. Conversely, keys
that a script never reads are simply ignored, and any new key added to the file 
becomes available on ``h``. The tables below therefore list example values, not 
defaults. The user can and should personalize the YAML parameter file to the 
main ``kine`` code. 

Parameters by scenario
----------------------

Keys defined by each parameter file shipped in ``parameters/``
(``params_static_imaging.yml``, ``params_multifreq_imaging.yml``,
``params_multiepoch_imaging.yml``, ``params_dynamic_imaging.yml``,
``params_dynamic_imaging_pol.yml``):

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 15 15

   * - Parameter
     - Static
     - Spectral
     - Multi-epoch
     - Dynamic
     - Dynamic pol.
   * - ``tavg``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``syserr``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``tflag``
     -
     -
     -
     - ✓
     - ✓
   * - ``min_bl``
     -
     -
     -
     - ✓
     - ✓
   * - ``fov_uas``
     - ✓
     - ✓
     - ✓
     - ``_0 _1 _2``
     - ✓
   * - ``npix``
     - ✓
     - ✓
     - ✓
     - ``_0 _1 _2``
     - ✓
   * - ``npix_out``
     - ✓
     - ✓
     - ✓
     -
     -
   * - ``data_prod``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``gains_prior``
     -
     -
     -
     - ✓
     -
   * - ``init_params``
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - ``seed``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``niter``
     - ✓
     - ✓
     - ✓
     - ``_0 _1 _2``
     - ✓
   * - ``initniter``
     - ✓
     - ✓
     - ✓
     - ``_0 _1 _2``
     - ✓
   * - ``nposenc``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``depth``
     - ✓
     - ✓
     - ✓
     - ``s_`` / ``d_``
     - ✓
   * - ``width``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``outshift``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``scaling_i``
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - ``scaling_ml``
     -
     -
     -
     -
     - ✓

Data Preprocessing
------------------

See also :doc:`user_guide` block 3.

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Type
     - Example
     - Description
   * - ``tavg``
     - float
     - ``60``
     - Coherent time-averaging interval in seconds, passed to
       :meth:`~kine.obsdata.Obsdata.avg_coherent`. Reduces data volume (and
       therefore memory and runtime) and raises the SNR per point. Set to
       ``0`` to leave the data unaveraged.
   * - ``syserr``
     - float
     - ``0.01``
     - Fractional systematic noise budget, passed to
       :meth:`~kine.obsdata.Obsdata.add_fractional_noise`. A value of ``0.01``
       adds 1% of the visibility amplitude in quadrature to the uncertainties.
   * - ``tflag``
     - dict
     - see below
     - UT-range flagging, passed to
       :meth:`~kine.obsdata.Obsdata.flag_UT_range`. Sub-keys ``t0`` (start UT
       hour), ``t1`` (stop UT hour) and ``out`` (``'kept'`` or ``'flagged'``).
   * - ``min_bl``
     - int
     - ``4``
     - Minimum number of **stations** required in a time snapshot, passed to
       :meth:`~kine.obsdata.Obsdata.split_obs` and
       :meth:`~kine.obsdata.Obsdata.get_lightcurve`. Snapshots with fewer are 
       dropped. Set to ``3`` when imaging with closure phases and ``4`` when 
       imaging with closure amplitudes; ``0`` disables the cut.

**Time flagging example:**

.. code-block:: yaml

   tflag:
     t0: 10.85      # window start, UT hours
     t1: 14.05      # window stop, UT hours
     out: flagged   # 'flagged' keeps the data inside [t0, t1]
                    # 'kept'    keeps the data outside [t0, t1]

.. Note::

   The ``out`` key follows ``ehtim``'s convention, in which the UT window is
   what gets *flagged*: ``output='kept'`` returns the data that survive the
   flagging, i.e. everything **outside** ``[t0, t1]``, while
   ``output='flagged'`` returns the data that were flagged, i.e. everything
   **inside** ``[t0, t1]``. To restrict an observation to a good UT window, as
   ``params_dynamic_imaging.yml`` does, use ``out: flagged``.


Coordinates and Resolution
--------------------------

See also :doc:`user_guide` block 4.

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Type
     - Example
     - Description
   * - ``fov_uas``
     - float
     - ``160``
     - Field of view in microarcseconds, converted to radians with
       ``eh.RADPERUAS`` and used both for the coordinate grid and for the
       ``improxy`` image metadata. In ``example_dynamic_imaging.py`` the
       indexed variants ``fov_uas_0``, ``fov_uas_1``, ``fov_uas_2`` give the
       field of view of each pipeline round.
   * - ``npix``
     - int
     - ``64``
     - Number of pixels per side of the training grid, so the pixel size is
       ``fov_uas / npix``. In ``example_dynamic_imaging.py`` the indexed
       variants ``npix_0``, ``npix_1``, ``npix_2`` give the resolution of each
       pipeline round.
   * - ``npix_out``
     - int
     - ``200``
     - Resolution the trained network is re-sampled at when the final image,
       video or cube is written (:doc:`user_guide` block 10).

.. note::

   Memory scales steeply with ``npix``: the DFT stores a dense
   ``(nvis × npix²)`` complex matrix per snapshot and per Fourier operator.
   Increase ``npix_out`` rather than ``npix`` when a finer output grid is all
   that is needed, and switch to the NUFFT when the DFT no longer fits in
   memory.


Data Products
-------------

See also :doc:`user_guide` block 5.

.. list-table::
   :header-rows: 1
   :widths: 18 10 20 52

   * - Parameter
     - Type
     - Example
     - Description
   * - ``data_prod``
     - list[str]
     - ``[logampI, cphaseI, logcampI]``
     - Data products entering the fit, as a list of string codes. One
       :math:`\chi^2` term is built per entry and all terms are summed with
       equal weight.

.. _data-product-codes:

**Data product codes**

Each code is a product name followed by a single letter naming the Stokes
parameter; the letter selects the network output channel the product is
computed from, so it must be one of ``I``, ``Q``, ``U``, ``V``. ``mbreve`` is
the one exception and carries no letter.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Code
     - Quantity
     - Notes
   * - ``visI``, ``visQ``, ``visU``, ``visV``
     - Complex visibilities :math:`V_{AB}`
     - Carry the full information content. The right choice when the data are
       well calibrated or have been self-calibrated.
   * - ``ampI``
     - Visibility amplitudes :math:`|V_{AB}|`
     - Immune to phase errors, sensitive to amplitude gains.
   * - ``logampI``
     - Log amplitudes :math:`\log|V_{AB}|`
     - As above, with better-behaved gradients over a wide dynamic range.
   * - ``cphaseI``
     - Closure phases :math:`\arg(V_{AB}V_{BC}V_{CA})`
     - Invariant under station-based phase errors. Carry no information on the
       absolute source position.
   * - ``logcampI``
     - Log closure amplitudes :math:`\log|\frac{V_{AB}V_{BC}}{V_{CD}V_{DA}}|`
     - Invariant under station-based amplitude errors. Carry no information on
       the total flux.
   * - ``bsI``
     - Bispectra :math:`V_{AB}V_{BC}V_{CA}`
     - Alternative to closure phases, retaining amplitude information.
   * - ``mbreve``
     - :math:`\breve m = (\tilde Q + i\tilde U)/\tilde I`
     - Complex polarization ratio, for polarimetric imaging.

A typical choice for EHT total-intensity data is
``[logampI, cphaseI, logcampI]``: closure quantities robust to station-based
calibration errors, plus log-amplitudes for absolute flux information. For
polarimetric imaging with Stokes I held fixed, use ``[visQ, visU]``.

``data_prod`` also sets the number of network output channels:

.. code-block:: python

   outdim = 1
   if 'visQ' in h.data_prod: outdim = 4
   if 'visV' in h.data_prod: outdim = 5

.. note::

   The NUFFT path is currently implemented for Stokes I only and supports
   ``visI``, ``ampI``, ``logampI``, ``cphaseI`` and ``logcampI``; bispectra,
   ``mbreve`` and simultaneous gain fitting require the DFT path.


Gain Fitting
------------

See :doc:`user_guide` block 8.

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Type
     - Example
     - Description
   * - ``gains_prior``
     - dict
     - see below
     - Per-station amplitude gain bounds, as ``[lower, upper]`` multiplicative
       factors. Read by :meth:`~kine.obsdata.Obsdata.set_gains_vars`.

**Gain prior example:**

.. code-block:: yaml

   gains_prior:
     AA: [0.97, 1.03]   # ALMA: well calibrated, tight bounds
     AP: [0.97, 1.03]   # APEX
     AZ: [0.90, 1.10]   # SMT
     JC: [0.97, 1.03]   # JCMT
     LM: [0.85, 1.15]   # LMT: known calibration issues, loose bounds
     SM: [0.97, 1.03]   # SMA
     SP: [0.94, 1.06]   # SPT

.. important::

   The station codes must match those in the array table of the UV-FITS file,
   and an entry is required for **every** station that survives flagging:
   :meth:`~kine.obsdata.Obsdata.set_gains_vars` looks up each station of the
   array table in ``gains_prior`` and raises a ``KeyError`` if one is missing.
   Give a station a range of ``[1.0, 1.0]`` to hold its gain fixed.

:class:`kine.model.AmplitudeGains` holds one amplitude gain per station and per
frame, initialized to ``1.0`` and clipped to ``[lower, upper]`` at every step.
:class:`kine.model.PhaseGains` holds one phase per station and per frame,
initialized to ``0.0`` and wrapped to :math:`[-\pi, \pi]`; it has no prior and
is therefore not configurable from the YAML file.


Network Initialization
----------------------

See :doc:`user_guide` block 7.

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Type
     - Example
     - Description
   * - ``init_params``
     - dict
     - see below
     - Geometry of the disk the network is pre-trained on, passed to
       :meth:`~kine.video.Video.add_tophat` /
       :meth:`~kine.video.Image.add_tophat`.

**Initialization example:**

.. code-block:: yaml

   init_params:
     fwhm: 80     # disk diameter in uas
     blur: 20     # circular Gaussian blurring in uas
     posx: 0      # horizontal offset in pixels; negative is left
     posy: 0      # vertical offset in pixels; negative is up

The disk flux is *not* set here: it is taken from the light curve (one value
per frame) or from the zero-baseline flux (static imaging). Only the geometry
matters, and only loosely — the initialization mainly serves to place the flux
in the centre of the frame when imaging with closure phases, and its detailed
shape does not affect the converged result.

``init_params`` is not used by ``example_dynamic_imaging_pol.py``, which
initializes from a previously reconstructed Stokes I video with
:meth:`~kine.video.Video.add_video_i` and
:meth:`~kine.video.Video.add_constant_linpol` instead.


Training
--------

See :doc:`user_guide` blocks 6, 7 and 9.

.. list-table::
   :header-rows: 1
   :widths: 18 10 14 58

   * - Parameter
     - Type
     - Example
     - Description
   * - ``seed``
     - int
     - ``1``
     - Random seed, used for both ``jax.random.PRNGKey`` and
       ``np.random.seed``. The same seed reproduces the same network
       initialization and therefore the same reconstruction; change it to
       explore the output variability due to different random initializations.
   * - ``initniter``
     - int
     - ``3000``
     - Number of initialization (pre-training) iterations, in which the
       network is regressed pixel-to-pixel onto the initialization image or
       video with no Fourier transform involved. It also sets the boundary of
       the ``optax.piecewise_constant_schedule`` learning-rate drop, so the
       large initialization learning rate is reduced exactly when data-driven
       training begins. Indexed variants ``initniter_0/1/2`` per pipeline
       round.
   * - ``niter``
     - int
     - ``10000``
     - Number of data-driven training iterations. Indexed variants
       ``niter_0/1/2`` per pipeline round. Also passed to the
       :class:`~kine.video.Video` / :class:`~kine.video.Image` constructor and,
       in dynamic imaging, used as the length of the gain learning-rate
       schedule.
   * - ``nposenc``
     - list[int]
     - ``[6, 0, 0]``
     - Degree of the Fourier-feature positional encoding, one entry per input
       coordinate, in the order the coordinate grid is built. See below.

**Positional encoding degrees**

``nposenc`` is passed to :class:`kine.model.NeuralField` as
``posenc_deg=tuple(h.nposenc)`` and expands each coordinate as

.. math::

   x \rightarrow \left[x, \sin(x), \cos(x), \ldots,
   \sin(2^{\mathrm{deg}}x), \cos(2^{\mathrm{deg}}x)\right].

A degree of ``0`` leaves that coordinate unencoded, so the spectral bias of the
MLP suppresses fast variation along it. Higher degrees make it easier for the
network to represent rapid variation along that coordinate.

The length of the list must match the number of columns of the coordinate
grid:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Scenario
     - Length
     - Columns
   * - Static imaging
     - 2
     - ``[x, y]``
   * - Dynamic / multi-epoch imaging
     - 3
     - ``[t, x, y]``
   * - Spectral imaging
     - 3
     - ``[f, x, y]``

Typical values are ``4``--``8`` for the time coordinate of an intra-track
dynamic reconstruction, and ``0`` for the spatial coordinates. In
``example_dynamic_imaging.py`` the same list serves both networks: the 3D
dynamic network receives ``tuple(h.nposenc)`` and the 2D static network
receives ``tuple(h.nposenc[-2:])``, i.e. the spatial entries only.


Network Architecture
--------------------

These keys are passed straight to :class:`kine.model.NeuralField` (or
:class:`kine.model.NeuralFieldPol`); see :doc:`user_guide` block 6.

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 14 48

   * - Parameter
     - Type
     - Example
     - Class default
     - Description
   * - ``depth``
     - int
     - ``6``
     - ``6``
     - Number of hidden layers. In ``example_dynamic_imaging.py`` the static
       and dynamic networks are sized separately by ``s_depth`` and
       ``d_depth``; the static network can be shallower (e.g. ``4``) because
       it represents a single 2D image.
   * - ``width``
     - int
     - ``256``
     - ``256``
     - Number of neurons per hidden layer.
   * - ``outshift``
     - int
     - ``10``
     - ``10``
     - Shift subtracted from the output logits before the output activation,
       ``outactiv(x - outshift)``. It pushes the initial, near-zero logits far
       into the flat tail of the softplus or sigmoid, so the network starts
       from an almost empty image and controls how quickly the output
       saturates.
   * - ``scaling_i``
     - float
     - ``1.0``
     - ``1.0``
     - Multiplicative scaling of the Stokes I channel,
       ``outactiv(x - outshift) * scaling_i``. With ``outactiv=nn.sigmoid``
       (the normalized components of the static + dynamic decomposition) it
       caps the per-pixel value at ``scaling_i``.
   * - ``scaling_ml``
     - float
     - ``0.75``
     - ``1.0``
     - Scaling of the linear polarization fraction channel,
       ``sigmoid(x - outshift) * scaling_ml``, i.e. the maximum fractional
       polarization the network can produce. Relevant whenever the network has
       polarization channels (:class:`~kine.model.NeuralField` with
       ``outdim >= 4``, or :class:`~kine.model.NeuralFieldPol`).

The remaining network arguments are set in the scripts rather than in the YAML
file: ``activ`` (``nn.gelu``, or :func:`kine.model.sharpgelu` through
``partial(mo.sharpgelu, s=3)``), ``outactiv`` (``nn.softplus`` for a single
network, ``nn.sigmoid`` for the normalized components of a decomposition),
``outdim`` (derived from ``data_prod``), ``do_bnorm`` and ``skipat``. The EVPA
channels use fixed sigmoids and are not configurable, and
:class:`~kine.model.NeuralFieldPol` applies a fixed sigmoid to its
:math:`m_\ell` channel regardless of ``outactiv``.


Example Parameter Files
-----------------------

The five files below are shipped in ``parameters/`` and are the ones referenced
by ``scripts/run_kine.sh``.

**Static imaging** (``params_static_imaging.yml``)

.. code-block:: yaml

   # Data pre-processing
   tavg: 60
   syserr: 0.01
   min_bl: 0

   # Coordinates and data products
   fov_uas: 160
   npix: 64
   npix_out: 200
   data_prod: [cphaseI, logcampI]

   # Network initialization
   init_params: {fwhm: 80, blur: 20, posx: 0, posy: 0}

   # Training
   seed: 1
   niter: 5000
   initniter: 2000
   nposenc: [0, 0]          # two entries: 2D grid

   # Network
   depth: 4
   width: 256
   outshift: 10
   scaling_i: 1

**Spectral imaging** (``params_multifreq_imaging.yml``)

.. code-block:: yaml

   tavg: 0
   syserr: 0.01
   min_bl: 0

   fov_uas: 100
   npix: 100
   npix_out: 200
   data_prod: [cphaseI, logcampI]

   init_params: {fwhm: 70, blur: 20, posx: 0, posy: 0}

   seed: 1
   niter: 2500
   initniter: 3000
   nposenc: [0, 0, 0]       # [f, x, y]

   depth: 4
   width: 256
   outshift: 10
   scaling_i: 1

**Multi-epoch imaging** (``params_multiepoch_imaging.yml``)

.. code-block:: yaml

   tavg: 0
   syserr: 0.01
   min_bl: 0

   fov_uas: 1000
   npix: 300                # training resolution
   npix_out: 300            # output (re-sampling) resolution
   data_prod: [cphaseI, logcampI]

   init_params: {fwhm: 60, blur: 20, posx: -50, posy: 50}

   seed: 1
   niter: 30000
   initniter: 3000
   nposenc: [4, 0, 0]       # [t, x, y]

   depth: 6
   width: 256
   outshift: 10
   scaling_i: 1

**Dynamic Stokes I imaging** (``params_dynamic_imaging.yml``)

.. code-block:: yaml

   # Data pre-processing
   tavg: 60
   syserr: 0.01
   tflag: {t0: 10.85, t1: 14.05, out: flagged}
   min_bl: 4

   # Multi-resolution pipeline
   fov_uas_0: 160
   fov_uas_1: 160
   fov_uas_2: 200
   npix_0: 16
   npix_1: 32
   npix_2: 64

   # Data products
   data_prod: [logampI, cphaseI, logcampI]

   # Gains (one entry per station in the array table)
   gains_prior:
     AA: [0.97, 1.03]
     AP: [0.97, 1.03]
     AZ: [0.90, 1.10]
     JC: [0.97, 1.03]
     LM: [0.85, 1.15]
     SM: [0.97, 1.03]
     SP: [0.94, 1.06]

   # Network initialization
   init_params: {fwhm: 80, blur: 20, posx: 0, posy: 0}

   # Training (one value per pipeline round)
   seed: 1
   niter_0: 10000
   niter_1: 10000
   niter_2: 5000
   initniter_0: 3000
   initniter_1: 3000
   initniter_2: 6000
   nposenc: [6, 0, 0]       # [t, x, y]; static network uses [0, 0]

   # Network (separate depths for static and dynamic fields)
   s_depth: 4
   d_depth: 6
   width: 256
   outshift: 10
   scaling_i: 1

**Dynamic polarimetric imaging** (``params_dynamic_imaging_pol.yml``)

.. code-block:: yaml

   tavg: 60
   syserr: 0.01
   tflag: {t0: 10.85, t1: 14.05, out: flagged}
   min_bl: 4

   fov_uas: 200
   npix: 64
   data_prod: [visQ, visU]  # Stokes I is held fixed, so it is not fitted

   seed: 1
   niter: 5000
   initniter: 3000
   nposenc: [4, 0, 0]       # [t, x, y]

   depth: 4
   width: 256
   outshift: 10
   scaling_ml: 0.75         # maximum linear polarization fraction
