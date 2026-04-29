
video
=====

The :class:`~kine.video.Video` class is the central container for
reconstructed image cubes. It bundles all Stokes/polarization arrays
together with their world-coordinate metadata, exposes constructors for
building a video from a Flax training state or from a saved file, and
provides plotting and export routines.

**Construction from training output**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.video.Video
   ~kine.video.Video.from_state
   ~kine.video.Video.from_states
   ~kine.video.Video.from_video
   ~kine.video.Video.from_h5

**Adding ancillary components**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.video.Video.add_tophat
   ~kine.video.Video.add_video_i
   ~kine.video.Video.add_constant_linpol
   ~kine.video.Video.add_constant_circpol

**Plotting**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.video.Video.plot
   ~kine.video.Video.plot_gif
   ~kine.video.Video.async_plot

**Saving and exporting**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.video.Video.save_gains
   ~kine.video.Video.save_fits
   ~kine.video.Video.save_h5

.. autoclass:: kine.video.Video
   :members:
   :undoc-members:
   :show-inheritance:

