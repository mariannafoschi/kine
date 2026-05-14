# CUDA Conda Environment

These instructions build a CUDA-enabled `kine` environment on a Linux machine
with a recent NVIDIA driver. They were written for a CUDA 12.8-capable server.

## Build the Environment

From the `kine` repository root:

```bash
conda env create -f environment-cuda.yml
conda activate kine
```
## Build jax-finufft With GPU Support

The public `jax-finufft` wheels may not include the GPU extension. Build it from
GitHub after the `kine` environment exists:

```bash
conda activate kine
git clone --recursive https://github.com/dfm/jax-finufft.git
cd jax-finufft

export JAX_FINUFFT_CUDA_ARCH=80  # A100; adjust for your GPU if needed.
export NANOBIND_DIR="$(python -m nanobind --cmake_dir)"
export CMAKE_ARGS="-DJAX_FINUFFT_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=${JAX_FINUFFT_CUDA_ARCH} \
  -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
  -Dnanobind_DIR=${NANOBIND_DIR} \
  -DFFTW_INCLUDE_DIRS=${CONDA_PREFIX}/include \
  -DFFTW_INCLUDE_DIR=${CONDA_PREFIX}/include \
  -DFFTW_LIBRARIES=${CONDA_PREFIX}/lib/libfftw3.so"

python -m pip install --force-reinstall --no-deps --no-build-isolation .
```

If CMake cannot find CUDA, set `CUDACXX` before the install command:

```bash
export CUDACXX=/path/to/nvcc
```

## Install and Test kine

Return to the `kine` repository:

```bash
cd /path/to/kine
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_kine_cuda.py --require-gpu --require-nufft
```

The smoke test imports the runtime stack, checks that JAX sees a GPU, compiles a
small `kine` model, and executes a tiny `jax-finufft` transform.
