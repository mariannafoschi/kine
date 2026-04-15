# Copyright (C) 2026 Antonio Fuentes

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Utils, helper functions, etc."""

import os
import threading
from collections.abc import Callable
from contextlib import contextmanager, redirect_stdout

import numpy as np

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


@contextmanager
def no_print():
    """Suppress stdout within the context."""
    with open(os.devnull, "w") as f:
        with redirect_stdout(f):
            yield

class HyperParams:
    """Create a class object to store hyperparameters."""
    def __init__(self, params: dict) -> None:
        self.__dict__.update(params)

class Schedule:
    """Create a learning rate schedule.

    Attributes:
        lr_i: Initial learning rate.
        lr_f: Final learning rate.
        niter: Number of training iterations.
    
    Todo:
        * Add more custom schedules
    """
    def __init__(self, lr_i: float, lr_f: float, niter: int) -> None:
        self.lr_i: float = lr_i
        self.lr_f: float = lr_f
        self.niter: int = niter
    
    def exponential(self, count: int) -> Array:
        """Exponential learning rate schedule.

        Args:
            count: Current iteration.

        Returns:
            Learning rate value at given iteration.
        """
        log_i = jnp.log(self.lr_i)
        log_f = jnp.log(self.lr_f)
        frac = jnp.clip(count / self.niter, 0.0, 1.0)
        log_lr = log_i + frac * (log_f - log_i)
        return jnp.exp(log_lr)

def init_worker(fn: Callable, *args) -> None:
    """Asynchronous worker for CPU plotting.

    Args:
        fn: Asynchronous plotting function.
        *args: Queue object.
    """
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()

def list_to_jaxarr(*args) -> Array | list[Array]:
    """Convert a list of arguments to JAX arrays."""
    if len(args) == 1:
        return jnp.array(args[0])
    return [jnp.array(arg) for arg in args]

def to_complex(arr: ArrayLike) -> Array:
    """Turn real-valued JAX array to complex type."""
    return jax.lax.complex(arr, jnp.zeros_like(arr))[..., None]

def stack_and_pad(arr: list[ArrayLike]) -> Array:
    """Stack and pad an inhomogeneous list to create an array.
        
    Args:
        arr: Inhomogeneous array list.
    
    Returns:
        Concatenated array with homogeneous shape.
    """
    # Stacking
    for i, _ in enumerate(arr):
        arr[i] = np.concatenate(arr[i])
    # Padding
    maxv = np.max([len(arr) for arr in arr])
    for i, _ in enumerate(arr):
        if len(arr[i]) < maxv:
            if len(arr[i].shape) > 1:
                arr[i] = np.concatenate(
                    [arr[i], np.ones((maxv-len(arr[i]), arr[i].shape[-1]))]
                )
            else:
                arr[i] = np.concatenate(
                    [arr[i], np.ones((maxv-len(arr[i]),))]
                )
    return jnp.array(arr)

def pad(arr: list[ArrayLike]) -> Array:
    """Pad an inhomogeneous list to create an array.
    
    Args:
        arr: Inhomogeneous array list.
    
    Returns:
        Concatenated array with homogeneous shape.
    """
    maxv = np.max([len(arr) for arr in arr])
    for i, _ in enumerate(arr):
        arr[i] = np.array(arr[i])
        if len(arr[i]) < maxv:
            if len(arr[i].shape) > 1:
                arr[i] = np.concatenate(
                    [arr[i], np.ones((maxv-len(arr[i]), arr[i].shape[-1]))]
                )
            else:
                arr[i] = np.concatenate(
                    [arr[i], np.ones((maxv-len(arr[i]),))]
                )
    return jnp.array(arr)

def map_val_to_ind(arr1: ArrayLike, arr2: ArrayLike) -> Array:
    """Map arr2 values to indices in arr1."""
    arr1 = arr1[:, :, None]
    arr2 = arr2[:, None, :]
    mask = arr1 == arr2
    return jnp.argmax(mask, axis=1)

def batchify(batch: list | ArrayLike, *args) -> list[Array] | Array:
    """Batch arrays.
    
    Args:
        batch: Sequence of indices for batching.
        *args: Arrays to batch.
    
    Returns:
        Batched arrays.
    """
    batched = []
    for arg in args:
        if arg is None:
            batched.append(None)
        elif isinstance(arg, dict):
            if any(isinstance(value, dict) for value in arg.values()):
                batched.append(
                    {key1: {
                        key2: arr[batch, ...]
                        for key2, arr in val1.items()
                    }
                    for key1, val1 in arg.items()
                    }
                )
            else:
                batched.append(
                    {key: arr[batch, ...] for key, arr in arg.items()}
                )
        else:
            batched.append(arg[batch, ...])
    return batched if len(batched) > 1 else batched[0]

def get_grid(
        nx: int,
        ny: int,
        nt: int | None = None,
        times: ArrayLike | None = None,
        tdil: float = 10
) -> Array:
    """Generate grid of space-time coordinates.

    The network is trained to predict the emission at locations (x, y, t)
    given by the initial grid of coordinates, but will learn a smooth
    interpolation between them that can be later sampled by passing a
    different (finer and time-homogeneous) grid of coordinates.

    Args:
        nx: Number of spatial locations in Right Ascension.
        ny: Number of spatial locations in Declination.
        nt: Number of time locations.
        times: Array of (irregular) time locations.
        tdil: Time dilation factor.
    
    Returns:
        Grid of space-time coordinates.
    """
    # 3D grid (t,x,y)
    if nt is not None:
        xx = jnp.linspace(0, 1, nx)
        yy = jnp.linspace(0, 1, ny)
        tt = jnp.linspace(0, 1, nt) / tdil
        if times is not None:
            tt = (times - times[0]) / (times[-1] - times[0]) / tdil
            nt = len(times)
        mesh = jnp.meshgrid(tt, xx, yy, indexing='ij')
        grid = jnp.stack(mesh, axis=-1)
        grid = grid.reshape(nt, -1, 3)
    # 2D grid (x,y)
    else:
        xx = np.linspace(0, 1, nx)
        yy = np.linspace(0, 1, ny)
        mesh = np.meshgrid(xx, yy, indexing='ij')
        grid = np.stack(mesh, axis=-1)
        grid = grid.reshape(-1, 2)
    return grid

def get_times_multiepoch(inpaths: str, ymd: bool = False) -> Array:
    """Extract observation times from multiepoch file paths.

    Args:
        inpaths: 
        ymd: If True, return times in YYYY-MM-DD format. If False,
            return times in mjd format (required for training).

    Returns:
        Array of dates.
    """
    from astropy.time import Time
    times = [os.path.basename(path).split('.')[-2] for path in inpaths]
    if ymd:
        return times
    return jnp.array([int(Time(time).mjd) for time in times])

def get_static_flux(
        found_flux: float,
        min_lcurve: float,
        min_flux_offset: float = 0.1
) -> float:
    """Determine static flux density.
    
    Args:
        found_flux: Flux density found through regularization.
        min_lcurve: Light-curve minimum value.
        min_flux_offset: Offset from light-curve minimum.
    """
    if found_flux < 0.95:
        if found_flux <= (min_lcurve - min_flux_offset):
            return found_flux
        return min_lcurve - min_flux_offset
    return found_flux
