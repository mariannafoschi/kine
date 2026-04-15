====
kine
====

**Interferometric video reconstruction with Neural Fields**

``kine`` is a Python package for video reconstruction of variable and sparse radio-interferometric data, from horizon-scale supermassive black holes to relativistic jets and more. It models the time-dependent brightness distribution of the observed source through a fully unsupervised *Neural Field*, or coordinate-based neural network.

Built on `JAX <https://jax.readthedocs.io/>`_ and `Flax <https://flax.readthedocs.io/>`_, ``kine`` leverages GPU-accelerated automatic differentiation and JIT compilation for fast training. It extends the `eht-imaging <https://github.com/achael/eht-imaging>`_ library for VLBI data handling.

.. image:: ../gif/kine_EHT.gif
   :align: center
   :alt: kine video reconstruction from EHT-like data

Features
--------

- **Full polarimetric** video and image reconstruction (Stokes I, Q, U, V).
- **Static + dynamic decomposition**: separate persistent and time-variable source structure.
- **Simultaneous gain fitting**: amplitude and phase telescope gains optimized jointly with the image.
- **Multi-epoch imaging**: reconstruct source evolution across observations spanning days to years.
- **GPU-based NUFFT**: optional Non-Uniform Fast Fourier Transform for direct visibility computation.
- **Multiple data products**: visibility amplitudes, closure phases, closure amplitudes, bispectra, and complex polarization ratios.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   user_guide
   parameters
   api
