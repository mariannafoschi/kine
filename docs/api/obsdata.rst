obsdata
=======

The :class:`~kine.obsdata.Obsdata` class extends `ehtim's Obsdata
<https://github.com/achael/eht-imaging/blob/main/ehtim/obsdata.py>`_ with
methods tailored to ``kine``'s neural-field training pipeline: flagging,
normalization, time splitting, light-curve extraction, and packing of
visibilities and closure quantities into JAX-friendly arrays.

**Loading and merging observations**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.obsdata.Obsdata.load_uvfits
   ~kine.obsdata.Obsdata.merge_obs

**Cleaning and flagging**

.. autosummary::
   :toctree: generated
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
   :toctree: generated
   :nosignatures:

   ~kine.obsdata.Obsdata.get_zbl
   ~kine.obsdata.Obsdata.norm_to_max
   ~kine.obsdata.Obsdata.fix_multiepoch

**Time splitting and light curves**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.obsdata.Obsdata.split_obs
   ~kine.obsdata.Obsdata.get_lightcurve

**Data product packing**

.. autosummary::
   :toctree: generated
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