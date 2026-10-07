trainer
=======

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
   ~kine.trainer.Trainer.create
   ~kine.trainer.Trainer.train_step
   ~kine.trainer.Trainer.apply_gradients
   ~kine.trainer.Trainer.replace

.. autodata:: kine.trainer.NPIX
   :no-value:

.. autoclass:: kine.trainer.Trainer
   :inherited-members: PyTreeNode

Loss terms
----------

:meth:`~kine.trainer.Trainer.train_step` does not take a loss function as an
argument. Instead it calls ``_loss_fn_red``, which delegates to
``_which_loss_fn``; that in turn picks a top-level loss from the *keys
present in* ``kwargs`` (``init_arr`` selects the initialization loss,
``uvpoints`` the NUFFT loss, ``s_grid`` the static + dynamic decomposition,
and so on). The chosen loss then assembles a ``chi2`` term per requested
data product through ``_loss_chi``, if requested adds the regularizer terms, and 
sums them.

Loss selection and top-level losses
...................................

.. autosummary::
   :nosignatures:

   ~kine.trainer.Trainer._loss_fn_red
   ~kine.trainer.Trainer._which_loss_fn
   ~kine.trainer.Trainer._loss_fn_init
   ~kine.trainer.Trainer._loss_fn_init_pol
   ~kine.trainer.Trainer._loss_fn
   ~kine.trainer.Trainer._loss_fn_pol
   ~kine.trainer.Trainer._loss_fn_nfft
   ~kine.trainer.Trainer._loss_fn_div_gains
   ~kine.trainer.Trainer._loss_fn_div_gains_fluxreg

.. automethod:: kine.trainer.Trainer._loss_fn_red
.. automethod:: kine.trainer.Trainer._which_loss_fn
.. automethod:: kine.trainer.Trainer._loss_fn_init
.. automethod:: kine.trainer.Trainer._loss_fn_init_pol
.. automethod:: kine.trainer.Trainer._loss_fn
.. automethod:: kine.trainer.Trainer._loss_fn_pol
.. automethod:: kine.trainer.Trainer._loss_fn_nfft
.. automethod:: kine.trainer.Trainer._loss_fn_div_gains
.. automethod:: kine.trainer.Trainer._loss_fn_div_gains_fluxreg

:math:`\chi^2` loss selection and gain correction
.................................................

.. autosummary::
   :nosignatures:

   ~kine.trainer.Trainer._loss_chi
   ~kine.trainer.Trainer._loss_gains

.. automethod:: kine.trainer.Trainer._loss_chi
.. automethod:: kine.trainer.Trainer._loss_gains

Data product loss terms
.......................

Each data product has up to three variants: ``_2d`` for static imaging,
``_3d`` for dynamic imaging (frame-padded, weighted by ``padmask``), and
``_3d_nfft`` for dynamic imaging with the NUFFT.

.. autosummary::
   :nosignatures:

   ~kine.trainer.Trainer._loss_vis_2d
   ~kine.trainer.Trainer._loss_vis_3d
   ~kine.trainer.Trainer._loss_vis_3d_nfft
   ~kine.trainer.Trainer._loss_amp_2d
   ~kine.trainer.Trainer._loss_amp_3d
   ~kine.trainer.Trainer._loss_amp_3d_nfft
   ~kine.trainer.Trainer._loss_logamp_2d
   ~kine.trainer.Trainer._loss_logamp_3d
   ~kine.trainer.Trainer._loss_logamp_3d_nfft
   ~kine.trainer.Trainer._loss_logcamp_2d
   ~kine.trainer.Trainer._loss_logcamp_3d
   ~kine.trainer.Trainer._loss_logcamp_3d_nfft
   ~kine.trainer.Trainer._loss_cphase_2d
   ~kine.trainer.Trainer._loss_cphase_3d
   ~kine.trainer.Trainer._loss_cphase_3d_nfft
   ~kine.trainer.Trainer._loss_bs_2d
   ~kine.trainer.Trainer._loss_bs_3d
   ~kine.trainer.Trainer._loss_mbreve_2d
   ~kine.trainer.Trainer._loss_mbreve_3d

.. automethod:: kine.trainer.Trainer._loss_vis_2d
.. automethod:: kine.trainer.Trainer._loss_vis_3d
.. automethod:: kine.trainer.Trainer._loss_vis_3d_nfft
.. automethod:: kine.trainer.Trainer._loss_amp_2d
.. automethod:: kine.trainer.Trainer._loss_amp_3d
.. automethod:: kine.trainer.Trainer._loss_amp_3d_nfft
.. automethod:: kine.trainer.Trainer._loss_logamp_2d
.. automethod:: kine.trainer.Trainer._loss_logamp_3d
.. automethod:: kine.trainer.Trainer._loss_logamp_3d_nfft
.. automethod:: kine.trainer.Trainer._loss_logcamp_2d
.. automethod:: kine.trainer.Trainer._loss_logcamp_3d
.. automethod:: kine.trainer.Trainer._loss_logcamp_3d_nfft
.. automethod:: kine.trainer.Trainer._loss_cphase_2d
.. automethod:: kine.trainer.Trainer._loss_cphase_3d
.. automethod:: kine.trainer.Trainer._loss_cphase_3d_nfft
.. automethod:: kine.trainer.Trainer._loss_bs_2d
.. automethod:: kine.trainer.Trainer._loss_bs_3d
.. automethod:: kine.trainer.Trainer._loss_mbreve_2d
.. automethod:: kine.trainer.Trainer._loss_mbreve_3d

Regularizers loss terms
.......................

.. autosummary::
   :nosignatures:

   ~kine.trainer.Trainer._loss_lcurve
   ~kine.trainer.Trainer._loss_min_dynamics
   ~kine.trainer.Trainer._loss_dynamic_flux
   ~kine.trainer.Trainer._loss_static_flux
   ~kine.trainer.Trainer._loss_border
   ~kine.trainer.Trainer._loss_ml_overlap

.. automethod:: kine.trainer.Trainer._loss_lcurve
.. automethod:: kine.trainer.Trainer._loss_min_dynamics
.. automethod:: kine.trainer.Trainer._loss_dynamic_flux
.. automethod:: kine.trainer.Trainer._loss_static_flux
.. automethod:: kine.trainer.Trainer._loss_border
.. automethod:: kine.trainer.Trainer._loss_ml_overlap
