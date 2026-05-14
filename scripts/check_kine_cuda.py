#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import platform
import sys


def require_import(module: str) -> object:
    try:
        return importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - diagnostic script
        raise SystemExit(f"FAILED import {module}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail unless JAX sees at least one GPU device",
    )
    parser.add_argument(
        "--require-nufft",
        action="store_true",
        help="fail unless jax-finufft imports and executes a tiny transform",
    )
    args = parser.parse_args()

    print(f"python: {sys.version.split()[0]} ({platform.platform()})")

    modules = [
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "yaml",
        "tqdm",
        "ehtim",
        "jax",
        "flax",
        "optax",
        "kine.model",
        "kine.obsdata",
        "kine.trainer",
        "kine.video",
        "kine.utils",
    ]
    for module in modules:
        imported = require_import(module)
        version = getattr(imported, "__version__", "unknown")
        print(f"import ok: {module} {version}")

    import jax
    import jax.numpy as jnp
    import kine.model as mo

    devices = jax.devices()
    print("jax devices:", ", ".join(str(device) for device in devices))
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if args.require_gpu and not gpu_devices:
        raise SystemExit("FAILED: --require-gpu was set, but JAX sees no GPU")

    @jax.jit
    def compiled_norm(x):
        return jnp.sqrt(jnp.sum(x * x))

    compiled_norm(jnp.arange(16.0)).block_until_ready()
    print("jax jit compile ok")

    key = jax.random.PRNGKey(0)
    grid = jnp.zeros((8, 3), dtype=jnp.float32)
    model = mo.NeuralField(posenc_deg=(1, 1, 1), depth=1, width=8, do_bnorm=False)
    variables = model.init(key, grid, train=True)
    out = model.apply(variables, grid, train=False)
    out.block_until_ready()
    print("kine NeuralField compile ok:", tuple(out.shape))

    try:
        from jax_finufft import nufft2
    except Exception as exc:
        if args.require_nufft:
            raise SystemExit(f"FAILED import jax_finufft: {exc}") from exc
        print(f"jax-finufft skipped: {exc}")
    else:
        image = jnp.ones((8, 8), dtype=jnp.complex64)
        u = jnp.linspace(-1.0, 1.0, 4, dtype=jnp.float32)
        v = jnp.linspace(-1.0, 1.0, 4, dtype=jnp.float32)
        vis = nufft2(image, v, u, eps=1e-3)
        vis.block_until_ready()
        print("jax-finufft nufft2 ok:", tuple(vis.shape))

    print("kine environment smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
