video
=====

The :class:`~kine.video.Video` class is the central container for reconstructed 
videos or spectral image cubes. It bundles all Stokes/polarization arrays
together with corresponding metadata, contains constructors for building videos 
from a Flax training state or from a saved file, and provides plotting and 
export routines.

The :class:`~kine.video.Image` class is its time-independent counterpart,
used for static imaging. It offers the same construction, plotting and
export interface for a single frame.

.. autosummary::
   :nosignatures:

   ~kine.video.Video
   ~kine.video.Image

Video
-----

**Construction from training output**

.. autosummary::
   :nosignatures:

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
   :exclude-members: __init__

Image
-----

**Construction from training output**

.. autosummary::
   :nosignatures:

   ~kine.video.Image.from_state
   ~kine.video.Image.from_image
   ~kine.video.Image.from_fits

**Adding ancillary components**

.. autosummary::
   :nosignatures:

   ~kine.video.Image.add_tophat
   ~kine.video.Image.add_image_i
   ~kine.video.Image.add_constant_linpol
   ~kine.video.Image.add_constant_circpol

**Plotting**

.. autosummary::
   :nosignatures:

   ~kine.video.Image.plot
   ~kine.video.Image.async_plot

**Saving and exporting**

.. autosummary::
   :nosignatures:

   ~kine.video.Image.save_fits

.. autoclass:: kine.video.Image
   :exclude-members: __init__
