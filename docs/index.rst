====
kine
====
(pronounciation: */ˈkine/*)

``kine`` is a Python package for video reconstruction of variable and sparse radio-interferometric data, from horizon-scale supermassive black holes to relativistic jets and more. It models the time-dependent brightness distribution of the observed source through a fully unsupervised neural field, parametrized by a coordinate-based neural network.

Built on `JAX <https://jax.readthedocs.io/>`_ and `Flax <https://flax.readthedocs.io/>`_, ``kine`` leverages GPU-accelerated automatic differentiation and JIT compilation for fast training. It extends the `eht-imaging <https://github.com/achael/eht-imaging>`_ library for VLBI data handling.

Imaging modes
-------------
- **Static imaging**: reconstruct an image of the source from a single VLBI observation.
- **Dynamic imaging**: reconstruct a video of the source from a single VLBI observation.
- **Multi-epoch imaging**: reconstruct a video of the source's evolution across multiple observations spanning days to years.

Available Features
------------------
- **Full polarimetric** video and image reconstruction (Stokes I, Q, U, V).
- **Static + dynamic decomposition**: in dynamic mode, separate persistent and time-variable source structure.
- **Simultaneous gain fitting**: amplitude and phase telescope gains optimized jointly with the image.
- **GPU-based NUFFT**: optional Non-Uniform Fast Fourier Transform for direct visibility computation.
- **Multiple data products**: visibility amplitudes, closure phases, closure amplitudes, bispectra, and complex polarization ratios.



.. image:: _images/kine_EHT.gif
   :align: center
   :alt: kine video reconstruction from EHT-like data


Available Features
------------------
.. toctree::
   :maxdepth: 2

   getting_started
   parameters
   user_guide
   api


