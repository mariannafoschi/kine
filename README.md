# [ REPOSITORY UNDER CONSTRUCTION ]

# kine

`kine` is a Python package for video reconstruction of variable and sparse radio-interferometric data, from horizon-scale supermassive black holes to relativistic jets and more. It models the time-dependent brightness distribution of the observed source through a fully unsupervised neural field, parametrized by a coordinate-based neural network.

Built on [`JAX`](https://jax.readthedocs.io/) and [`Flax`](https://flax.readthedocs.io/), ``kine`` leverages GPU-accelerated automatic differentiation and JIT compilation for fast training. It extends the [`eht-imaging`](https://github.com/achael/eht-imaging) library for VLBI data handling.

#### Imaging modes

`kine` can be used for the following imaging tasks:

- **Static imaging**: reconstruct an image of the source from a single VLBI observation.
- **Dynamic imaging**: reconstruct a video of the source from a single VLBI observation.
- **Multi-epoch imaging**: reconstruct a video of the source's evolution across multiple observations spanning days to years.

#### Available features

`kine` currently supports:

- **Full polarimetric** video and image reconstruction (Stokes I, Q, U, V).
- **Static + dynamic decomposition**: in dynamic mode, separate persistent and time-variable source structure.
- **Simultaneous gain fitting**: amplitude and phase telescope gains optimized jointly with the image.
- **GPU-based NUFFT**: Non-Uniform Fast Fourier Transform for direct visibility computation.
- **Multiple data products**: visibility amplitudes, closure phases, closure amplitudes, bispectra, and complex polarization ratios.



![kine video reconstruction from EHT-like data](docs/images/kine_EHT.gif)



## Installation

`kine` relies on the `JAX` library for GPU computations and requires a careful installation of CUDA-related packages and others. For reference, a working conda environment can be found in [environment.yml](https://github.com/aefezeta/kine/tree/main/environment.yml). Detailed instructions on the installation will be provided in the near future.

Assuming you have all required dependencies already installed, then install `kine` from the root directory with:

    $ pip install -e .

## Documentation

Full documentation is available at (https://mariannafoschi.github.io/kine/) and includes quick start example scripts, description of diagnostic plots, and API documentation.

A full description of the imaging algorithm and extensive reconstruction validation tests are presented in the publications:

1. Foschi, M., Zhao, B., Fuentes, A. et al. "Video reconstruction of variable interferometric observations with neural fields." Under rev. (2026).
2. Fuentes, A., Foschi, M. et al. "Validation of horizon-scale Sagittarius A* video reconstructions with kine" In prep. (2026).

## Developers

`kine` is developed and maintained by:

 - Antonio Fuentes (antoniofuentesfdez @ gmail . com)
 - Marianna Foschi (foschimarianna @ gmail . com)
 - Brandon Zhao (byzhao @ caltech . edu)

 ## Citation

 If you use `kine` in your publication, please cite:

1. Foschi, M., Zhao, B., Fuentes, A. et al. "Video reconstruction of variable interferometric observations with neural fields." Under rev. (2026).
2. Fuentes, A., Foschi, M. et al. "Validation of horizon-scale Sagittarius A* video reconstructions with kine" In prep. (2026).
