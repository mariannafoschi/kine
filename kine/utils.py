# Copyright (C) 2026 Marianna Foschi, Antonio Fuentes, Brandon Zhao

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

def _mjd_from_uvfits(path: str) -> float:
    """Read the observation MJD from the metadata of a uvfits file.

    Follows the same convention as ``ehtim``: the observation date is the
    smallest Julian date stored in the ``DATE`` random group parameter(s),
    which are optionally rescaled by the ``PSCAL``/``PZERO`` keywords. Files
    that are not in random-groups format fall back on the ``DATE-OBS``
    header keyword.

    Args:
        path: Path to a uvfits file.

    Returns:
        Observation date in (fractional) mjd format.
    """
    from astropy.io import fits
    from astropy.time import Time

    with fits.open(path) as hdulist:
        header = hdulist['PRIMARY'].header
        data = hdulist['PRIMARY'].data
        parnames = list(getattr(data, 'parnames', []))
        if 'DATE' not in parnames:
            if 'DATE-OBS' not in header:
                raise ValueError(
                    f'No DATE parameter or DATE-OBS keyword found in {path}. '
                    'Pass a file name format through `fmt` instead.'
                )
            return float(Time(header['DATE-OBS']).mjd)

        # First DATE parameter (1-indexed, as in the PSCAL/PZERO keywords)
        idx = parnames.index('DATE') + 1
        jds = (
            header.get(f'PSCAL{idx}', 1.) * np.asarray(data['DATE'], dtype='d')
            + header.get(f'PZERO{idx}', 0.)
        )
        # Second DATE parameter (holding the fraction of a day), if present
        if parnames.count('DATE') > 1:
            jds = jds + (
                header.get(f'PSCAL{idx + 1}', 1.)
                * np.asarray(data['_DATE'], dtype='d')
                + header.get(f'PZERO{idx + 1}', 0.)
            )
    return float(np.min(jds) - 2400000.5)

def _mjd_from_filename(path: str, fmt: str) -> float:
    """Extract the observation MJD from a file name.

    Args:
        path: Path to an observation file.
        fmt: ``datetime.strptime`` format matching the whole file name,
            e.g. ``'obs_%Y_%m_%d.uvfits'``.

    Returns:
        Observation date in (fractional) mjd format.
    """
    from datetime import datetime

    from astropy.time import Time

    name = os.path.basename(path)
    return float(Time(datetime.strptime(name, fmt)).mjd)

def get_times_multiepoch(
        inpaths: str | list,
        ymd: bool = False,
        fmt: str | None = None,
        integer: bool = True
) -> Array | list:
    """Extract observation times from multiepoch observations.

    By default the times are read from the metadata of each uvfits file, so
    no assumption is made on how the files are named. Alternatively, a list
    of already loaded ``ehtim.obsdata.Obsdata`` objects can be passed (their
    ``mjd`` attribute is used), or the times can be parsed from the file
    names by passing a format through `fmt`.

    Args:
        inpaths: List of paths to the observation files (or a single path),
            or list of ``Obsdata`` objects.
        ymd: If True, return times in YYYY-MM-DD format. If False,
            return times in mjd format (required for training).
        fmt: Optional ``datetime.strptime`` format matching the whole file
            name (e.g. ``'obs_%Y_%m_%d.uvfits'``), used to parse the dates
            from the file names instead of reading them from the metadata.
        integer: If True, round the times down to integer mjd (one time
            coordinate per day). Ignored if `ymd` is True.

    Returns:
        Array of mjd times, or list of YYYY-MM-DD strings if `ymd` is True.
    """
    from astropy.time import Time

    if isinstance(inpaths, str):
        inpaths = [inpaths]

    mjds = []
    for path in inpaths:
        if fmt is not None:
            mjds.append(_mjd_from_filename(path, fmt))
        elif hasattr(path, 'mjd'):  # already loaded Obsdata object
            mjds.append(float(path.mjd) + float(getattr(path, 'time', 0.)) / 24)
        else:
            mjds.append(_mjd_from_uvfits(path))

    if ymd:
        return [Time(mjd, format='mjd').iso[:10] for mjd in mjds]
    if integer:
        return jnp.array([int(mjd) for mjd in mjds])
    return jnp.array(mjds)

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
