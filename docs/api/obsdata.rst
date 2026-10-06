obsdata
=======

The :class:`~kine.obsdata.Obsdata` class extends
:class:`ehtim.obsdata.Obsdata` with methods tailored to ``kine``'s
neural-field training pipeline: flagging, normalization, time splitting,
light-curve extraction, and packing of visibilities and closure quantities
into JAX-friendly arrays.

Only the ``kine``-specific additions are listed below. Every attribute and
method of the ``ehtim`` base class remains available; follow the ``Bases:``
link at the bottom of this page for those.

**Loading and merging observations**

.. autosummary::
   :nosignatures:

   ~kine.obsdata.Obsdata.load_uvfits
   ~kine.obsdata.Obsdata.merge_obs

**Flux and time-split utilities**

.. autosummary::
   :nosignatures:

   ~kine.obsdata.Obsdata.get_zbl
   ~kine.obsdata.Obsdata.get_lightcurve
   ~kine.obsdata.Obsdata.norm_to_max
   ~kine.obsdata.Obsdata.fix_multiepoch
   ~kine.obsdata.Obsdata.fix_multifreq
   ~kine.obsdata.Obsdata.split_obs

**Preprocessing**

.. autosummary::
   :nosignatures:

   ~kine.obsdata.Obsdata.flag_empty
   ~kine.obsdata.Obsdata.flag_UT_range
   ~kine.obsdata.Obsdata.flag_uvdist
   ~kine.obsdata.Obsdata.flag_sites
   ~kine.obsdata.Obsdata.flag_bl
   ~kine.obsdata.Obsdata.avg_coherent
   ~kine.obsdata.Obsdata.add_fractional_noise

**Data information extraction**

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
