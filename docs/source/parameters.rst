===================
Parameter Reference
===================

``kine`` uses YAML configuration files to specify imaging parameters. This page documents all available parameters with their types, defaults, and descriptions.


Data Preprocessing
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Parameter
     - Type
     - Example
     - Description
   * - ``tavg``
     - float
     - ``60``
     - Coherent time-averaging interval in seconds. Reduces data volume and noise. Set to 0 to disable.
   * - ``syserr``
     - float
     - ``0.01``
     - Fractional systematic noise added to the data. A value of 0.01 adds 1% of the visibility amplitude as additional uncertainty.
   * - ``tflag``
     - dict
     - see below
     - Time flagging configuration. Sub-keys: ``t0`` (start UT hour), ``t1`` (end UT hour), ``out`` (``'kept'`` or ``'flagged'``).
   * - ``min_bl``
     - int
     - ``4``
     - Minimum number of baselines required per time snapshot. Snapshots with fewer baselines are discarded.

**Time flagging example:**

.. code-block:: yaml

   tflag:
     t0: 10.85
     t1: 14.05
     out: flagged  # 'flagged' keeps data inside window; 'kept' keeps outside


Coordinates and Data Products
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Parameter
     - Type
     - Example
     - Description
   * - ``fov_uas`` / ``fov_uas_0``, ``fov_uas_1``, ``fov_uas_2``
     - float
     - ``160``
     - Field of view in microarcseconds. Indexed variants (``_0``, ``_1``, ``_2``) set per-step values in the multi-step pipeline.
   * - ``npix`` / ``npix_0``, ``npix_1``, ``npix_2``
     - int
     - ``64``
     - Image resolution in pixels (per side). Indexed variants set per-step values.
   * - ``data_prod``
     - list
     - ``[logampI, cphaseI, logcampI]``
     - Data products to fit. See :ref:`data-product-codes` below.

.. _data-product-codes:

**Data product codes:**

Each code consists of a product type followed by a Stokes parameter letter:

- ``visI``, ``visQ``, ``visU``, ``visV`` --- Complex visibilities
- ``ampI`` --- Visibility amplitudes
- ``logampI`` --- Log visibility amplitudes
- ``cphaseI`` --- Closure phases
- ``logcampI`` --- Log closure amplitudes
- ``bsI`` --- Bispectra
- ``mbreve`` --- Complex linear polarization ratio (Q + iU) / I

A typical choice for EHT data is ``[logampI, cphaseI, logcampI]``, which uses closure quantities robust to station-based calibration errors plus log-amplitudes for absolute flux information.


Gain Fitting
------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Parameter
     - Type
     - Example
     - Description
   * - ``gains_prior``
     - dict
     - see below
     - Per-telescope amplitude gain bounds as ``[lower, upper]`` multiplicative factors.

**Gain prior example:**

.. code-block:: yaml

   gains_prior:
     AA: [0.97, 1.03]   # ALMA: well-calibrated, tight bounds
     AP: [0.97, 1.03]   # APEX
     AZ: [0.90, 1.10]   # SMT
     JC: [0.97, 1.03]   # JCMT
     LM: [0.85, 1.15]   # LMT: known calibration issues, loose bounds
     SM: [0.97, 1.03]   # SMA
     SP: [0.94, 1.06]   # SPT

Telescope codes must match those in the UV-FITS file. Gains are initialized to 1.0 (amplitude) or 0.0 (phase) and clipped to the specified bounds during training.


Network Initialization
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Parameter
     - Type
     - Example
     - Description
   * - ``init_params``
     - dict
     - see below
     - Parameters for the disk initialization model.

**Initialization example:**

.. code-block:: yaml

   init_params:
     fwhm: 80     # Disk diameter in microarcseconds
     blur: 20     # Gaussian blurring in microarcseconds
     posx: 0      # Horizontal position offset (pixels). Negative = left.
     posy: 0      # Vertical position offset (pixels). Negative = up.


Training
--------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Parameter
     - Type
     - Example
     - Description
   * - ``seed``
     - int
     - ``1``
     - Random seed for reproducibility (JAX and NumPy).
   * - ``niter`` / ``niter_0``, ``niter_1``, ``niter_2``
     - int
     - ``10000``
     - Number of data-driven training iterations per step.
   * - ``initniter`` / ``initniter_0``, ``initniter_1``, ``initniter_2``
     - int
     - ``3000``
     - Number of initialization (pre-training) iterations per step.
   * - ``nposenc``
     - list[int]
     - ``[6, 0, 0]``
     - Positional encoding degrees for ``[t, x, y]``. Higher temporal values capture finer time variability. Typical range: 4--8 for time, 0 for space.


Network Architecture
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Parameter
     - Type
     - Example
     - Description
   * - ``depth`` / ``s_depth``, ``d_depth``
     - int
     - ``6``
     - Number of hidden layers. ``s_depth`` and ``d_depth`` set separate depths for the static and dynamic networks. The static network can typically be shallower (e.g., 4).
   * - ``width``
     - int
     - ``256``
     - Number of neurons per hidden layer.
   * - ``outshift``
     - int
     - ``10``
     - Shift applied before the output activation function. Controls how quickly output saturates from the initial near-zero state.
   * - ``scaling_i``
     - float
     - ``1.0``
     - Stokes I output scaling factor. The output activation is multiplied by this value.
   * - ``scaling_ml``
     - float
     - ``0.75``
     - Linear polarization fraction output scaling. Sets the maximum allowed polarization fraction (relevant for polarimetric imaging).


Example Parameter Files
-----------------------

**Dynamic Stokes I imaging** (``dynamic_imaging_params.yml``):

.. code-block:: yaml

   # Data preprocessing
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

   # Gains
   gains_prior:
     AA: [0.97, 1.03]
     LM: [0.85, 1.15]

   # Initialization
   init_params: {fwhm: 80, blur: 20, posx: 0, posy: 0}

   # Training
   seed: 1
   niter_0: 10000
   niter_1: 10000
   niter_2: 5000
   initniter_0: 3000
   initniter_1: 3000
   initniter_2: 6000
   nposenc: [6, 0, 0]

   # Network
   s_depth: 4
   d_depth: 6
   width: 256
   outshift: 10
   scaling_i: 1

**Polarimetric imaging** (``dynamic_imaging_pol_params.yml``):

.. code-block:: yaml

   tavg: 60
   syserr: 0.01
   tflag: {t0: 10.85, t1: 14.05, out: flagged}
   min_bl: 4
   fov_uas: 200
   npix: 64
   data_prod: [visQ, visU]
   seed: 1
   niter: 5000
   initniter: 3000
   nposenc: [4, 0, 0]
   depth: 4
   width: 256
   outshift: 10
   scaling_ml: 0.75

**Multi-epoch imaging** (``multiepoch_imaging_params.yml``):

.. code-block:: yaml

   tavg: 0
   syserr: 0.01
   min_bl: 0
   fov_uas: 1000
   npix_1: 300
   npix_2: 300
   data_prod: [cphaseI, logcampI]
   init_params: {fwhm: 60, blur: 20, posx: -50, posy: 50}
   seed: 1
   niter: 30000
   initniter: 3000
   nposenc: [4, 0, 0]
   depth: 6
   width: 256
   outshift: 10
   scaling_i: 1
