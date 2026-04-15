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

"""Neural fields, learnable parameters, and related models."""

from typing import Any
from collections.abc import Callable
from functools import partial

import numpy as np

from flax import linen as nn
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike


def sharpgelu(x: ArrayLike, s: float = 3.0) -> Array:
    """Custom GELU activation function.

    Computes a `sharper` variant of the (approximate)
    Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x: input array.
        s: sharpness factor.
    
    Returns:
        The sharpGELU activation function.
    """
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
    cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (s*x + 0.044715 * x**3)))
    return x * cdf

def posenc(x: ArrayLike, degs: tuple[int]) -> Array:
    """Positional encoding.

    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig. identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
        x: Variable to be encoded. Note that x should be in [-pi, pi].
        deg: The degree of the encoding.

    Returns:
        Encoded variables.
    """
    def safe_sin(x):
        return jnp.sin(x % (100 * jnp.pi))
    out = [x]
    for j, _ in enumerate(degs):
        if degs[j] == 0:
            continue
        scales = jnp.array([2**i for i in range(degs[j])])
        xb = jnp.reshape((x[..., None, j] * scales), list(x.shape[:-1]) + [-1])
        sins = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
        out.append(sins)
    return jnp.concatenate(out, axis=-1)

class NeuralField(nn.Module):
    """Neural Field implementation.
    
    Predicts full pol. emission at space-time coordinates (x, y, t).
    Input coordinates are transformed through poisitional encoding
    and fed to an MLP. We use batch normalization and skip connections.
    
    Args:
        posenc_deg: Degrees of positional encoding
        outdim: Output layer dimension
        depth: Number of hidden layers
        width: Number of neurons in each hidden layer
        activ: Activation function for hidden layers
        outactiv: Output activation function
        outshift: Output activation function shift
        do_bnorm: If true, do batch normalization
        skipat: Skip connection layer
        scaling_i: Sigmoid scaling for Stokes I ([0, 1] -> [0, scaling_i])
        scaling_ml: Sigmoid scaling for ml ([0, 1] -> [0, scaling_ml])
    """

    posenc_deg: tuple[int] = (0, 0, 0)
    outdim: int = 1
    depth: int = 6
    width: int = 256
    activ: Callable[..., Any] = nn.gelu
    outactiv: Callable[..., Any] = nn.softplus
    outshift: int = 10
    do_bnorm: bool = True
    skipat: int = 0
    scaling_i: float = 1.0
    scaling_ml: float = 1.0

    @nn.compact
    def __call__(self, x: ArrayLike, train: bool) -> Array:
        """Applies the Neural Field to the input grid of coordinates.

        Args:
            x: Grid of space-time coordinates.
            train: If true, compute batch norm statistics using runnin average.

        Returns:
            Predicted full pol. emission at coordinates (x, y, t).
        """
        # Layer creation functions
        dense_layer = partial(
            nn.Dense,
            kernel_init=nn.initializers.he_uniform()
        )
        bnorm_layer = partial(nn.BatchNorm, use_running_average=not train)
        # Positional Encoding
        x = posenc(x, self.posenc_deg)
        # Init skip connection
        skip = 0
        # Multi Layer Perceptron
        for i in range(self.depth+1):
            x = dense_layer(self.width)(x)
            if self.do_bnorm:
                x = bnorm_layer()(x)
            x = self.activ(x)
            # Save output for skip connection
            if i == self.skipat:
                skip += x
        # Add skip connection and final output layer
        x += skip
        x = dense_layer(self.outdim)(x)
        # Rescale outputs with final activations, where
        # x[...,0] = Stokes I
        # x[...,1] = linear pol. fraction ml
        # x[...,2] = sin(2xi)
        # x[...,3] = cos(2xi)
        # x[...,4] = circular pol. fraction mc
        x = x.at[..., 0].set(
            self.outactiv(x[..., 0] - self.outshift) * self.scaling_i
        )
        if self.outdim >= 4:
            x = x.at[..., 1].set(
                nn.sigmoid(x[..., 1] - self.outshift) * self.scaling_ml
            )
            x = x.at[..., 2].set((nn.sigmoid(x[..., 2]) - 0.5) * 2)
            x = x.at[..., 3].set((nn.sigmoid(x[..., 3]) - 0.5) * 2)
        if self.outdim == 5:
            x = x.at[..., 4].set(
                (nn.sigmoid(x[..., 4] - self.outshift) - 0.5) * 2
            )
        return x

class NeuralFieldPol(nn.Module):
    """Linear polarization Neural Field implementation.
    
    Predicts lin. pol. emission at space-time coordinates (x, y, t).
    Input coordinates are transformed through poisitional encoding
    and fed to an MLP. We use batch normalization and skip connections.
    
    Args:
        posenc_deg: Degrees of positional encoding
        outdim: Output layer dimension
        depth: Number of hidden layers
        width: Number of neurons in each hidden layer
        activ: Activation function for hidden layers
        outactiv: Output activation function
        outshift: Output activation function shift
        do_bnorm: If true, do batch normalization
        skipat: Skip connection layer
        scaling_ml: Sigmoid scaling for ml ([0, 1] -> [0, scaling_ml])
    """

    posenc_deg: tuple[int] = (0, 0, 0)
    outdim: int = 3
    depth: int = 6
    width: int = 256
    activ: Callable[..., Any] = nn.gelu
    outactiv: Callable[..., Any] = nn.sigmoid
    outshift: int = 10
    do_bnorm: bool = True
    skipat: int = 0
    scaling_ml: float = 1.0

    @nn.compact
    def __call__(self, x: ArrayLike, train: bool) -> Array:
        """Applies the Neural Field to the input grid of coordinates.

        Args:
            x: Grid of space-time coordinates.
            train: If true, compute batch norm statistics using runnin average.

        Returns:
            Predicted lin. pol. emission at coordinates (x, y, t).
        """
        # Layer creation functions
        dense_layer = partial(
            nn.Dense,
            kernel_init=nn.initializers.he_uniform()
        )
        bnorm_layer = partial(nn.BatchNorm, use_running_average=not train)
        # Positional Encoding
        x = posenc(x, self.posenc_deg)
        # Init skip connection
        skip = 0
        # Multi Layer Perceptron
        for i in range(self.depth+1):
            x = dense_layer(self.width)(x)
            if self.do_bnorm:
                x = bnorm_layer()(x)
            x = self.activ(x)
            # Save output for skip connection
            if i == self.skipat:
                skip += x
        # Add skip connection and final output layer
        x += skip
        x = dense_layer(self.outdim)(x)
        # Rescale outputs with final activations, where
        # x[...,0] = linear pol. fraction ml
        # x[...,1] = sin(2xi)
        # x[...,2] = cos(2xi)
        x = x.at[..., 0].set(
            nn.sigmoid(x[..., 0] - self.outshift) * self.scaling_ml
        )
        x = x.at[..., 1].set((nn.sigmoid(x[..., 1]) - 0.5) * 2)
        x = x.at[..., 2].set((nn.sigmoid(x[..., 2]) - 0.5) * 2)
        return x

class AmplitudeGains(nn.Module):
    """Train a multi-dimensional parameter as amplitude gains.
    
    Args:
        lower: Gains values lower limit.
        upper: Gains values upper limit.
        nsite: Number of telescopes.
        ntimes: Number of time segments over which gains are computed.
    """

    lower: ArrayLike
    upper: ArrayLike
    nsites: int = 8
    ntimes: int = 99

    def clipping(self, x: ArrayLike, site: str) -> Array:
        """Clip gains within specified range.
        
        Args:
            x: Input array.
            site: Telescope codename.

        Returns:
            Clipped (or Sigmoided) input array.
        """
        lower = self.lower[site]
        upper = self.upper[site]
        # return lower + (upper - lower * jax.nn.sigmoid(x)
        return jnp.clip(x, lower, upper)

    @nn.compact
    def __call__(
        self,
        baselines: ArrayLike,
        frames: ArrayLike
    ) -> tuple[Array]:
        """Compute amplitude gains
        
        Args:
            baselines: Baselines indices for each time segment.
            frames: Sequence of frame numbers.
        
        Returns:
            Amplitude gains for sites i and j forming a baseline b_ij.
        """
        # Initialize gains to 1
        gains = self.param(
            'gains',
            lambda *_: jnp.ones((self.nsites, self.ntimes))
        )
        # Select telescopes indices and corresponding gains
        i = baselines[frames, :, 0]
        j = baselines[frames, :, 1]
        gi = self.clipping(gains[i, frames.reshape(-1, 1)], i)
        gj = self.clipping(gains[j, frames.reshape(-1, 1)], j)
        return gi, gj

class PhaseGains(nn.Module):
    """Train a multi-dimensional parameter as phase gains.
    
    Args:
        nsite: Number of telescopes.
        ntimes: Number of time segments over which gains are computed.
    """

    nsites: int = 8
    ntimes: int = 99

    def clipping(self, x: ArrayLike) -> Array:
        """Clip gains within specified range.
                
        Args:
            x: Input array.

        Returns:
            Clipped input array.
        """
        x = (x + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.clip(x, -jnp.pi, jnp.pi)

    @nn.compact
    def __call__(
        self,
        baselines: ArrayLike,
        frames: ArrayLike
    ) -> tuple[Array]:
        """Compute phase gains
        
        Args:
            baselines: Baselines indices for each time segment.
            frames: Sequence of frame numbers.
        
        Returns:
            Phase gains for sites i and j forming a baseline b_ij.
        """
        # Initialize gains to 0
        gains = self.param(
            'gains',
            lambda *_: jnp.zeros((self.nsites, self.ntimes))
        )
        # Select telescopes indices and corresponding gains
        i = baselines[frames, :, 0]
        j = baselines[frames, :, 1]
        gi = self.clipping(gains[i, frames.reshape(-1, 1)])
        gj = self.clipping(gains[j, frames.reshape(-1, 1)])
        return gi, gj

class _ComplexGains(nn.Module):
    """Train a multi-dimensional parameter as complex gains.

    Fitting for amplitude + phase gains
    separately seems to work better currently.
    
    Args:
        lower: Gain lower value allowed.
        upper: Gain upper value allowed.
        nsite: Number of telescopes.
        ntimes: Number of time segments over which gains are computed.
    """

    lower: ArrayLike
    upper: ArrayLike
    nsites: int = 8
    ntimes: int = 99

    def clipping(self, x, site):
        """Clip gains within specified range"""
        lower  = self.lower[site]
        upper  = self.upper[site]
        # return lower + (upper - lower) * jax.nn.sigmoid(x / 1)
        return jnp.clip(x, lower, upper)

    def clipping2(self, x):
        """Clip gains within specified range"""
        # return jnp.clip(x, -np.pi, np.pi)
        return jnp.mod(x, 2*np.pi) - np.pi

    @nn.compact
    def __call__(self, vis, baselines, frames):
        """Apply complex gain corrections to visibilities"""
        # Initialize gains to 1
        gains = self.param(
            'gains',
            lambda *_: jnp.ones((self.nsites, self.ntimes),
                                dtype=jnp.complex64)
        )
        # Select telescopes indices and corresponding gains
        i = baselines[frames, :, 0]
        j = baselines[frames, :, 1]
        mod_gi = self.clipping(jnp.abs(gains[i, frames.reshape(-1, 1)]), i)
        mod_gj = self.clipping(jnp.abs(gains[j, frames.reshape(-1, 1)]), j)
        arg_gi = self.clipping2(jnp.angle(gains[i, frames.reshape(-1, 1)]))
        arg_gj = self.clipping2(jnp.angle(gains[j, frames.reshape(-1, 1)]))
        gi = mod_gi * jnp.exp(1j * arg_gi)
        gj = mod_gj * jnp.exp(1j * arg_gj)
        # Apply gains and return modified amplitudes
        return gi * jnp.conj(gj) * vis

class _StaticFluxDensity(nn.Module):
    """Train a single parameter as the static flux density.

    It doesn't seem to work currently, but probably worth keeping it around.

    Args:
        init_value: Static flux density ([0, 1]) initial guess.
    """

    init_value: float = 0.5

    def setup(self):
        """Convert initial value to logit space."""
        init_logit = jnp.log(self.init_value) - jnp.log(1.0 - self.init_value)
        self.logit = self.param(
            "static_flux_density",
            lambda key, shape: jnp.array(init_logit),
            ()
        )

    def __call__(self):
        """Apply a Sigmoid to keep it within [0, 1]."""
        return nn.sigmoid(self.logit)
