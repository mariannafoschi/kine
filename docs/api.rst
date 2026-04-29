=============
API Reference
=============

.. currentmodule:: kine

This page provides a complete reference for the public ``kine`` API. The
package is organized into five modules, each documented in its own section
below:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :ref:`kine.model <api-model>`
     - Neural fields and learnable parameter modules.
   * - :ref:`kine.obsdata <api-obsdata>`
     - Data handling, extending ``ehtim``'s ``Obsdata`` class.
   * - :ref:`kine.trainer <api-trainer>`
     - Training state and training-step driver.
   * - :ref:`kine.video <api-video>`
     - Video object, plotting, and export utilities.
   * - :ref:`kine.utils <api-utils>`
     - Helpers for grids, batching, schedules, and I/O.


.. _api-model:

model --- Neural Fields and Learnable Parameters
-------------------------------------------------

.. automodule:: kine.model

**Activation and encoding helpers**

.. autosummary::
   :nosignatures:

   ~kine.model.sharpgelu
   ~kine.model.posenc

**Neural fields**

.. autosummary::
   :nosignatures:

   ~kine.model.NeuralField
   ~kine.model.NeuralFieldPol

**Telescope gain modules**

.. autosummary::
   :nosignatures:

   ~kine.model.AmplitudeGains
   ~kine.model.PhaseGains

.. autofunction:: kine.model.sharpgelu

.. autofunction:: kine.model.posenc

.. autoclass:: kine.model.NeuralField
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: kine.model.NeuralFieldPol
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: kine.model.AmplitudeGains
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: kine.model.PhaseGains
   :members:
   :undoc-members:
   :show-inheritance:


.. _api-obsdata:

obsdata --- Data Handling
-------------------------

.. automodule:: kine.obsdata

The :class:`~kine.obsdata.Obsdata` class extends `ehtim's Obsdata
<https://github.com/achael/eht-imaging/blob/main/ehtim/obsdata.py>`_ with
methods tailored to ``kine``'s neural-field training pipeline: flagging,
normalization, time splitting, light-curve extraction, and packing of
visibilities and closure quantities into JAX-friendly arrays.

**Loading and merging observations**

.. autosummary::
   :nosignatures:

   ~kine.obsdata.Obsdata.load_uvfits
   ~kine.obsdata.Obsdata.merge_obs

**Cleaning and flagging**

.. autosummary::
   :nosignatures:

   ~kine.obsdata.Obsdata.flag_empty
   ~kine.obsdata.Obsdata.flag_UT_range
   ~kine.obsdata.Obsdata.flag_uvdist
   ~kine.obsdata.Obsdata.flag_sites
   ~kine.obsdata.Obsdata.flag_bl
   ~kine.obsdata.Obsdata.avg_coherent
   ~kine.obsdata.Obsdata.add_fractional_noise

**Normalization and multi-epoch alignment**

.. autosummary::
   :nosignatures:

   ~kine.obsdata.Obsdata.get_zbl
   ~kine.obsdata.Obsdata.norm_to_max
   ~kine.obsdata.Obsdata.fix_multiepoch

**Time splitting and light curves**

.. autosummary::
   :nosignatures:

   ~kine.obsdata.Obsdata.split_obs
   ~kine.obsdata.Obsdata.get_lightcurve

**Data product packing**

.. autosummary::
   :nosignatures:

   ~kine.obsdata.Obsdata.get_data
   ~kine.obsdata.Obsdata.get_data_nfft
   ~kine.obsdata.Obsdata.get_baselines_nfft
   ~kine.obsdata.Obsdata.get_uvpoints
   ~kine.obsdata.Obsdata.get_pulsefac
   ~kine.obsdata.Obsdata.get_closure_baselines
   ~kine.obsdata.Obsdata.get_closure_indices
   ~kine.obsdata.Obsdata.set_gains_vars

.. autoclass:: kine.obsdata.Obsdata
   :members:
   :undoc-members:
   :show-inheritance:


.. _api-trainer:

trainer --- Training and Loss Functions
---------------------------------------

.. automodule:: kine.trainer

The :class:`~kine.trainer.Trainer` class extends Flax's
``train_state.TrainState`` with batch-norm statistics and bundles all
loss functions used during training as static methods. End users typically
only need :meth:`~kine.trainer.Trainer.create` (inherited from Flax) to
build a training state and :meth:`~kine.trainer.Trainer.train_step` to
advance it by one optimization step.

**Module-level globals**

.. autosummary::
   :nosignatures:

   ~kine.trainer.NPIX

**Training state**

.. autosummary::
   :nosignatures:

   ~kine.trainer.Trainer
   ~kine.trainer.Trainer.train_step

.. autodata:: kine.trainer.NPIX
   :annotation:

.. autoclass:: kine.trainer.Trainer
   :members:
   :undoc-members:
   :show-inheritance:


.. _api-video:

video --- Video Creation and Visualization
------------------------------------------

.. automodule:: kine.video

The :class:`~kine.video.Video` class is the central container for
reconstructed image cubes. It bundles all Stokes/polarization arrays
together with their world-coordinate metadata, exposes constructors for
building a video from a Flax training state or from a saved file, and
provides plotting and export routines.

**Construction from training output**

.. autosummary::
   :nosignatures:

   ~kine.video.Video
   ~kine.video.Video.from_state
   ~kine.video.Video.from_states
   ~kine.video.Video.from_video
   ~kine.video.Video.from_h5

**Adding ancillary components**

.. autosummary::
   :nosignatures:

   ~kine.video.Video.add_tophat
   ~kine.video.Video.add_video_i
   ~kine.video.Video.add_constant_linpol
   ~kine.video.Video.add_constant_circpol

**Plotting**

.. autosummary::
   :nosignatures:

   ~kine.video.Video.plot
   ~kine.video.Video.plot_gif
   ~kine.video.Video.async_plot

**Saving and exporting**

.. autosummary::
   :nosignatures:

   ~kine.video.Video.save_gains
   ~kine.video.Video.save_fits
   ~kine.video.Video.save_h5

.. autoclass:: kine.video.Video
   :members:
   :undoc-members:
   :show-inheritance:


.. _api-utils:

utils --- Utilities
-------------------

.. automodule:: kine.utils

**Hyperparameters and learning-rate schedules**

.. autosummary::
   :nosignatures:

   ~kine.utils.HyperParams
   ~kine.utils.Schedule

**Coordinate grids and time handling**

.. autosummary::
   :nosignatures:

   ~kine.utils.get_grid
   ~kine.utils.get_times_multiepoch
   ~kine.utils.get_static_flux

**Array helpers**

.. autosummary::
   :nosignatures:

   ~kine.utils.list_to_jaxarr
   ~kine.utils.to_complex
   ~kine.utils.stack_and_pad
   ~kine.utils.pad
   ~kine.utils.map_val_to_ind
   ~kine.utils.batchify

**I/O and concurrency**

.. autosummary::
   :nosignatures:

   ~kine.utils.no_print
   ~kine.utils.init_worker

.. autoclass:: kine.utils.HyperParams
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: kine.utils.Schedule
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: kine.utils.get_grid

.. autofunction:: kine.utils.get_times_multiepoch

.. autofunction:: kine.utils.get_static_flux

.. autofunction:: kine.utils.list_to_jaxarr

.. autofunction:: kine.utils.to_complex

.. autofunction:: kine.utils.stack_and_pad

.. autofunction:: kine.utils.pad

.. autofunction:: kine.utils.map_val_to_ind

.. autofunction:: kine.utils.batchify

.. autofunction:: kine.utils.no_print

.. autofunction:: kine.utils.init_worker
