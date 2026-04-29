trainer --- Training and Loss Functions
---------------------------------------

The :class:`~kine.trainer.Trainer` class extends Flax's
``train_state.TrainState`` with batch-norm statistics and bundles all
loss functions used during training as static methods. End users typically
only need :meth:`~kine.trainer.Trainer.create` (inherited from Flax) to
build a training state and :meth:`~kine.trainer.Trainer.train_step` to
advance it by one optimization step.

**Module-level globals**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.trainer.NPIX

**Training state**

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~kine.trainer.Trainer
   ~kine.trainer.Trainer.train_step

.. autodata:: kine.trainer.NPIX
   :annotation:

.. autoclass:: kine.trainer.Trainer
   :members:
   :undoc-members:
   :show-inheritance: