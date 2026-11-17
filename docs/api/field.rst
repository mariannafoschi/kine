field
=====

The :class:`~kine.field.Video` class is the central container for reconstructed 
videos or spectral image cubes. It bundles all Stokes/polarization arrays
together with corresponding metadata, contains constructors for building videos 
from a Flax training state or from a saved file, and provides plotting and 
export routines.

The :class:`~kine.field.Image` class is its time-independent counterpart,
used for static imaging. It offers the same construction, plotting and
export interface for a single frame.

.. autosummary::
   :nosignatures:

   ~kine.field.Video
   ~kine.field.Image

Video
-----

**Video object creation**

.. autosummary::
   :nosignatures:

   ~kine.field.Video.from_state
   ~kine.field.Video.from_states
   ~kine.field.Video.from_video
   ~kine.field.Video.from_h5

**Adding video arrays**

.. autosummary::
   :nosignatures:

   ~kine.field.Video.add_tophat
   ~kine.field.Video.add_video_i
   ~kine.field.Video.add_constant_linpol
   ~kine.field.Video.add_constant_circpol

**Plotting**

.. autosummary::
   :nosignatures:

   ~kine.field.Video.plot
   ~kine.field.Video.plot_gif
   ~kine.field.Video.async_plot

**Saving and exporting**

.. autosummary::
   :nosignatures:

   ~kine.field.Video.save_gains
   ~kine.field.Video.save_fits
   ~kine.field.Video.save_h5

.. autoclass:: kine.field.Video
   :exclude-members: __init__

Image
-----

**Image object creation**

.. autosummary::
   :nosignatures:

   ~kine.field.Image.from_state
   ~kine.field.Image.from_image
   ~kine.field.Image.from_fits

**Adding image arrays**

.. autosummary::
   :nosignatures:

   ~kine.field.Image.add_tophat
   ~kine.field.Image.add_image_i
   ~kine.field.Image.add_constant_linpol
   ~kine.field.Image.add_constant_circpol

**Plotting**

.. autosummary::
   :nosignatures:

   ~kine.field.Image.plot
   ~kine.field.Image.async_plot

**Saving and exporting**

.. autosummary::
   :nosignatures:

   ~kine.field.Image.save_fits

.. autoclass:: kine.field.Image
   :exclude-members: __init__
