model
=====

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
   ~kine.model.AmplitudeGains.clipping_ag
   ~kine.model.PhaseGains.clipping_pg

**Activation and encoding helpers**

.. autosummary::
   :nosignatures:

   ~kine.model.posenc
   ~kine.model.sharpgelu

.. autoclass:: kine.model.NeuralField

.. autoclass:: kine.model.NeuralFieldPol

.. autoclass:: kine.model.AmplitudeGains

.. autoclass:: kine.model.PhaseGains

.. autofunction:: kine.model.posenc
   
.. autofunction:: kine.model.sharpgelu
