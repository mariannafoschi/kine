utils
=====

**Hyperparameters and learning-rate schedules**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.utils.HyperParams
   ~kine.utils.Schedule

**Coordinate grids and time handling**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.utils.get_grid
   ~kine.utils.get_times_multiepoch
   ~kine.utils.get_static_flux

**Array helpers**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.utils.list_to_jaxarr
   ~kine.utils.to_complex
   ~kine.utils.stack_and_pad
   ~kine.utils.pad
   ~kine.utils.map_val_to_ind
   ~kine.utils.batchify

**I/O and concurrency**

.. autosummary::
   :toctree: generated
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
