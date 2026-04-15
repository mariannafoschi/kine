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

"""Video object creation, plotting and related tasks."""

from typing import Any
from collections.abc import Callable

import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

from jax import numpy as jnp
from jax.typing import ArrayLike

from . import utils as ut


class Video:
    """Creation, plotting, etc. of video arrays in full pol.

    Aims to have all video arrays organized under one common object
    to improve code readability and overall clearness. Plotting routines
    are also methods of this class, which makes the code significantly
    cleaner. Saving and exporting methods are as well included here.

    Attributes:
        times: UT time in hours assigned to each video frame.
        dates: YYYY-MM-DD date assigned to each video frame.
            Used for multi-epoch dynamic imaging.
        npix: Number of pixels.
        fov: Field of view in radians.
        ra: Right ascension in fractional hours.
        dec: Declination in fractional degrees.
        niter: Current number of iterations. Mainly for plotting.
        polchan: Number of polarization channels.
        iarr: Stokes I video array.
        larr: Lin. pol. frac. (ml) video array.
        xarr: EVPA (xi) video array.
        carr: Circ. pol. frac. (mc) video array.
        qarr: Stokes Q video array.
        uarr: Stokes U video array.
        varr: Stokes V video array.
        parr: Lin. pol. intensity video array.
        loss: Current loss value. Mainly for plotting.
        amp_gains: Fitted amplitude gains. Mainly for plotting.
    
    Todo:
        * Remove arrays extra dimension prev. used for polchan
        * Solve Matplotlib's exception when threading (harmless)
        * Fix MJDs
    """

    def __init__(
            self,
            times: ArrayLike,
            npix: int,
            fov: float,
            ra: float,
            dec: float,
            niter: int,
            dates: list[str] | None = None
    ) -> None:
        """Initialize class attributes.

        Args:
            times: UT time in hours assigned to each video frame.
            npix: Number of pixels.
            fov: Field of view in radians.
            ra: Right ascension in fractional hours.
            dec: Declination in fractional degrees.
            niter: Current number of iterations.
            dates: YYYY-MM-DD date assigned to each video frame.
                Defaults to None.
        """
        self.times: ArrayLike = times
        self.dates: list[str] | None = dates
        self.npix: int = npix
        self.fov: float = fov
        self.ra: float = ra
        self.dec: float = dec
        self.niter: int = niter
        self.polchan: int = 1
        self.iarr: ArrayLike | None = None
        self.larr: ArrayLike | None = None
        self.xarr: ArrayLike | None = None
        self.carr: ArrayLike | None = None
        self.qarr: ArrayLike | None = None
        self.uarr: ArrayLike | None = None
        self.varr: ArrayLike | None = None
        self.parr: ArrayLike | None = None
        self.loss: list[float] | dict | None = None
        self.amp_gains: dict | None = None

    def add_tophat(self, lcurve: ArrayLike, params: dict) -> None:
        """Add a flat disk to each frame in the video.

        Args:
            lcurve: Light-curve array for the time-variable disk flux density.
            params: Diks size ('fwhm'), blurring ('blur', in uas), and
                x, y position shift ('posx', 'posy', in pixel units).
        """
        # Create disk image
        disk = eh.image.make_empty(self.npix, self.fov, self.ra, self.dec)
        disk = disk.add_tophat(1, params['fwhm']*eh.RADPERUAS/2)
        disk = disk.blur_circ(params['blur']*eh.RADPERUAS)
        disk = disk.shift([params['posy'], params['posx']])
        # Repeat for each frame with corresponding light curve flux
        frames = []
        for i, _ in enumerate(self.times):
            disk.imvec *= lcurve[i] / disk.total_flux()
            frames.append(disk.imarr())
        # Make JAX array with shape (times, npix, npix, 1)
        self.iarr = jnp.array(frames)[..., jnp.newaxis]

    def add_video_i(self, inpath: str) -> None:
        """Load Stokes I video.

        Args:
            inpath: Path to h5 file.
        """
        video = eh.movie.load_hdf5(inpath).im_list()
        video = [
            frame.regrid_image(self.fov, self.npix).imarr()
            for frame in video
        ]
        self.iarr = jnp.array(video).reshape(-1, self.npix, self.npix, 1)

    def add_constant_linpol(
            self,
            linpolfrac: float = 0.2,
            evpa: float = -1.0
    ) -> None:
        """Add constant polarization to each frame in the video.

        Args:
            linpolfrac: Linear polarization fraction value.
            evpa: EVPA value.
        """
        self.polchan = 3
        self.larr = linpolfrac * jnp.ones(
            (len(self.times), self.npix, self.npix, 1)
        )
        self.xarr = evpa * jnp.ones(
            (len(self.times), self.npix, self.npix, 1)
        )

    def add_constant_circpol(self, circpolfrac: float = 0.05) -> None:
        """Add constant polarization to each frame in the video.

        Args:
            circpolfrac: Circular polarization fraction value.
        """
        self.polchan = 4
        self.carr = circpolfrac * jnp.ones(
            (len(self.times), self.npix, self.npix, 1)
        )

    def from_state(
            self,
            state: Callable,
            grid: ArrayLike,
            loss: list[float] | dict | None = None
    ) -> None:
        """Create Video object from current network's state.

        Intended for re-sampling on a different grid of
        space-time coordinates.
        
        Args:
            state: Current network's state.
            grid: Input space-time coordinates.
            loss: Current loss value.
        """
        out, _ = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            grid, train=True, mutable=['batch_stats']
        )
        polchan = out.shape[-1]
        self.iarr = out[..., 0].reshape(-1, self.npix, self.npix, 1)
        if polchan > 1:
            self.larr = out[..., 1].reshape(-1, self.npix, self.npix, 1)
            self.xarr = jnp.arctan2(
                out[..., 2], out[..., 3]
            ).reshape(-1, self.npix, self.npix, 1) * 0.5
            self.qarr = - self.iarr * self.larr * jnp.sin(2*self.xarr)
            self.uarr = self.iarr * self.larr * jnp.cos(2*self.xarr)
            self.parr = jnp.sqrt(self.qarr**2 + self.uarr**2)
        if polchan > 4:
            self.carr = out[..., 4].reshape(-1, self.npix, self.npix, 1)
            self.varr = self.iarr * self.carr
        self.loss = loss

    def from_states(
            self,
            s_state: Callable,
            d_state: Callable,
            s_grid: ArrayLike,
            d_grid: ArrayLike,
            lcurve: ArrayLike,
            min_lcurve: float,
            loss: list[float] | dict | None = None,
            amp_gains: dict | None = None
    ) -> None:
        """Create Video object from current networks' states.

        Intended for re-sampling on a different grid of
        space-time coordinates.
        
        Args:
            s_state: Current static network state.
            d_state: Current dynamic network state.
            s_grid: Input space coordinates for static network.
            d_grid: Input space-time coordinates for dynamic network.
            lcurve: Light-curve flux density array.
            min_lcurve: Static component flux density. It usually
                corresponds to the minimum value of the data light-curve.
            loss: Current loss value.
            amp_gains: Fitted visibility amplitude gains.
        """
        static, _ = s_state.apply_fn(
            {'params': s_state.params, 'batch_stats': s_state.batch_stats},
            s_grid, train=True, mutable=['batch_stats']
        )
        dynamic, _ = d_state.apply_fn(
            {'params': d_state.params, 'batch_stats': d_state.batch_stats},
            d_grid, train=True, mutable=['batch_stats']
        )
        mod_lcurve = lcurve - min_lcurve
        mod_lcurve = mod_lcurve.reshape(-1, 1, 1)
        out = static * min_lcurve + dynamic * mod_lcurve
        polchan = out.shape[-1]
        self.iarr = out[..., 0].reshape(-1, self.npix, self.npix, 1)
        if polchan > 1:
            self.larr = out[..., 1].reshape(-1, self.npix, self.npix, 1)
            self.xarr = jnp.arctan2(
                out[..., 2], out[..., 3]
            ).reshape(-1, self.npix, self.npix, 1) * 0.5
            self.qarr = - self.iarr * self.larr * jnp.sin(2*self.xarr)
            self.uarr = self.iarr * self.larr * jnp.cos(2*self.xarr)
            self.parr = jnp.sqrt(self.qarr**2 + self.uarr**2)
        if polchan > 4:
            self.carr = out[..., 4].reshape(-1, self.npix, self.npix, 1)
            self.varr = self.iarr * self.carr
        self.loss = loss
        self.amp_gains = amp_gains

    def from_video(
            self,
            out: ArrayLike,
            loss: list[float] | dict | None = None,
            amp_gains: dict | None = None
    ) -> None:
        """Create Video object from video array.

        Args:
            out: Video array sampled from the network.
            loss: Current loss value.
            amp_gains: Fitted visibility amplitude gains.
        """
        polchan, dim = out.shape[-1], 1
        if polchan != 3:
            dim -= 1
            self.iarr = out[..., 0].reshape(-1, self.npix, self.npix, 1)
        if polchan > 1:
            self.polchan = 3
            self.larr = out[..., 1-dim].reshape(-1, self.npix, self.npix, 1)
            self.xarr = jnp.arctan2(
                out[..., 2-dim], out[..., 3-dim]
            ).reshape(-1, self.npix, self.npix, 1) * 0.5
            self.qarr = - self.iarr * self.larr * jnp.sin(2*self.xarr)
            self.uarr = self.iarr * self.larr * jnp.cos(2*self.xarr)
            self.parr = jnp.sqrt(self.qarr**2 + self.uarr**2)
        if polchan > 4:
            self.polchan = 4
            self.carr = out[..., 4-dim].reshape(-1, self.npix, self.npix, 1)
            self.varr = self.iarr * self.carr
        self.loss = loss
        self.amp_gains = amp_gains

    def from_h5(
            self,
            inpath: str,
            blur: float = 0,
            fn: Callable | None = None
    ) -> None:
        """Create Video object from input h5 file.

        Note:
            Currently it only supports Stokes I.

        Args:
            inpath: Input path of h5 file.
            blur: Blurring factor in uas.
            fn: Averaging function (np.mean, np.median, etc.).
        """
        init = eh.movie.load_hdf5(inpath).im_list()
        init = [im.regrid_image(self.fov, self.npix) for im in init]
        if blur > 0: #TODO: before or after fn?
            init = [im.blur_circ(blur*eh.RADPERUAS) for im in init]
        init = [im.imarr() for im in init]
        if fn is not None:
            init = [fn(init, axis=0) for _ in init]
        init = [im/im.sum() for im in init]
        init = jnp.array(init) if len(self.times) > 1 else jnp.array([init[0]])
        self.iarr = init.reshape(-1, self.npix, self.npix, 1)

    def plot(
            self,
            s_out: ArrayLike | None = None,
            d_out: ArrayLike | None = None,
            scale: str = 'lin',
            drange: float = 1e3,
            vstep: int = 3,
            vscale: float = 0.05,
            show: bool = False,
            outpath: str = './tmp.png'
    ) -> None:
        """Plot frames (and current loss) from Video object in full pol.

        General plotting function for static and dynamic imaging.
        If fitting for linear polarization, pol. field vectors,
        lin. pol. frac., and EVPA plots will be shown along Stokes I.
        If training, loss progress will be shown as well.
        ...

        Args:
            s_out: Static component video array.
            d_out: Dynamic component video array.
            scale: 'lin' for linear color scale, 'log' for logarithmic.
            drange: Image dynamic range. Used when scale='log'.
            vstep: Pol. field vectors' spacing.
            vscale: Pol. field vectors' scaling factor.
            show: If true, show interactive plot.
            outpath: Path and filename for figure saving.
        """
        # Preamble
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 14
        plt.rcParams['image.interpolation'] = 'bicubic'
        # Extent
        extent = [
            self.fov/2/eh.RADPERUAS,
            -self.fov/2/eh.RADPERUAS,
            -self.fov/2/eh.RADPERUAS,
            self.fov/2/eh.RADPERUAS
        ]
        # Scale bar
        barl = self.fov / eh.RADPERUAS / 5
        # Set colorbar scale
        def normalize(arr, scale=scale, drange=drange):
            vmin, vmax = arr.min(), arr.max()
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            if scale == 'log':
                vmin = vmax / drange
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            return vmin, vmax, norm
        
        # Single dynamic network polarimetric plotting
        if s_out is None:
            # Select frames to show
            if len(self.times) >= 8:
                nrow, ncol = 2, 4
                frames = np.linspace(0, len(self.times)-1, 8, dtype=int)
            elif len(self.times) > 4 and len(self.times) < 8:
                nrow, ncol = 1, 4
                frames = np.linspace(0, len(self.times)-1, 4, dtype=int)
            else:
                nrow, ncol = 1, len(self.times) + int(self.loss is not None)
                frames = np.arange(len(self.times), dtype=int)
            # Titles, colormaps, etc
            cmap = ['inferno', 'viridis', 'twilight']
            label = ['Stokes I', 'Lin. pol. frac.', 'EVPA']
            # Subfiguring
            fig = plt.figure(figsize=(5*ncol, ((5*nrow)+1)*self.polchan))
            subfigs = fig.subfigures(self.polchan, 1)
            subfigs = np.atleast_1d(subfigs)
            # Plotting
            for j, subfig in enumerate(subfigs):
                ax = subfig.subplots(nrow, ncol)
                ax = np.atleast_1d(ax).ravel()
                if j == 0:
                    arr = self.iarr
                elif j == 1:
                    arr = self.larr
                else:
                    arr = self.xarr
                vmin, _, norm = normalize(arr)
                for i, f in enumerate(frames):
                    # Stokes I, ml, and EVPA
                    ax[i].imshow(
                        arr[f] + vmin,
                        extent=extent,
                        norm=norm,
                        cmap=cmap[j]
                    )
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                    # Labels
                    c = 'k' if j == 2 else 'w'
                    ax[i].text(
                        0.98,
                        0.98,
                        f'#{f}',
                        c=c,
                        ha='right',
                        va='top',
                        fontsize=14,
                        transform=ax[i].transAxes
                    )
                    if j == 0:
                        ax[i].text(
                            1,
                            -0.01,
                            f'$S_{{tot}}$: {arr[f].sum():.1f} Jy',
                            c='k',
                            ha='right',
                            va='top',
                            fontsize=12,
                            transform=ax[i].transAxes
                        )
                    if not isinstance(self.dates, list):
                        ax[i].text(
                            0.5,
                            1.02,
                            f'{self.times[f]:.2f} UT',
                            c='k',
                            ha='center',
                            va='bottom',
                            fontsize=16,
                            transform=ax[i].transAxes
                        )
                    else:
                        ax[i].text(
                            0.5,
                            1.02,
                            f'{self.dates[f]}',
                            c='k',
                            ha='center',
                            va='bottom',
                            fontsize=16,
                            transform=ax[i].transAxes
                        )
                    # Pol. field vectors on top of Stokes I
                    if self.qarr is not None and j == 0:
                        x = np.linspace(extent[0], extent[1], self.npix)
                        y = np.linspace(extent[0], extent[1], self.npix)
                        vx = np.sin(
                            np.angle(self.qarr[f] + 1j * self.uarr[f]) / 2
                        ) * self.parr[f] * -1
                        vy = np.cos(
                            np.angle(self.qarr[f] + 1j * self.uarr[f]) / 2
                        ) * self.parr[f]
                        ax[i].quiver(
                            x[::vstep],
                            y[::vstep],
                            vx[::vstep, ::vstep, 0],
                            vy[::vstep, ::vstep, 0],
                            self.larr[f, ::vstep, ::vstep, 0],
                            pivot='mid',
                            angles='uv',
                            width=0.01,
                            scale=vscale,
                            headwidth=0,
                            headlength=0,
                            headaxislength=0,
                            cmap='viridis'
                        )
                # Labels
                ax[0].text(
                    0,
                    1.08,
                    f'{label[j]}',
                    fontsize=16,
                    weight='bold',
                    c='k',
                    ha='left',
                    va='bottom',
                    transform=ax[0].transAxes
                )
                ax[0].text(
                    0.1,
                    -0.03,
                    f'{round(barl):1d} $\mu$as',
                    c='k',
                    ha='center',
                    va='top',
                    transform=ax[0].transAxes
                )
                ax[0].hlines(
                    -0.02,
                    0,
                    0.2,
                    transform=ax[0].transAxes,
                    colors='k',
                    lw=3,
                    clip_on=False
                )
                # Loss progress
                if self.loss is not None and j == 0:
                    ax[-1].clear()
                    if isinstance(self.loss, dict):
                        for l in self.loss:
                            ax[-1].plot(
                                np.linspace(
                                    0,
                                    len(self.loss[l]),
                                    len(self.loss[l][::50])
                                ),
                                self.loss[l][::50],
                                label=f'{l} = {self.loss[l][-1]:.3e}'
                            )
                            ax[-1].axhline(1, c='k', lw=0.5, zorder=0)
                    else:
                        ax[-1].plot(
                            np.linspace(
                                0,
                                len(self.loss),
                                len(self.loss[::50])
                            ),
                            self.loss[::50],
                            label=f'loss = {self.loss[-1]:.3e}'
                        )
                    ax[-1].set_yscale('log')
                    ax[-1].set_xlim(-self.niter/50, self.niter+self.niter/50)
                    ax[-1].set_xlabel('iterations')
                    ax[-1].yaxis.tick_right()
                    ax[-1].grid(alpha=0.5)
                    ax[-1].set_aspect('auto')
                    ax[-1].set_adjustable('box')
                    ax[-1].set_box_aspect(1)
                    ax[-1].legend(
                        loc='upper right',
                        bbox_to_anchor=(1, 1),
                        fontsize=12
                    )
                # Adjust plots spacing
                subfig.subplots_adjust(hspace=-0.05, wspace=0.05)
            fig.tight_layout()

        # Static + Dynamic network plotting
        else:
            # Figure
            fig, ax = plt.subplot_mosaic(
                '''
                abcdef
                ......
                ghijkl
                ......
                mnopqr
                ......
                stuvwx
                ''',
                figsize=(25, 30),
                height_ratios=[1, -0.795, 1, -0.75, 1, -0.795, 1]
            )
            # Select frames to show
            frames = np.linspace(0, len(self.times)-1, 12, dtype=int)
            # Normalization
            f_vmax, d_vmax = self.iarr.max(), d_out.max()
            vmin = 0
            # Full video frames
            ax['a'].text(
                0,
                1.15,
                'Full video',
                c='k',
                weight='bold',
                fontsize=16,
                transform=ax['a'].transAxes
            )
            for i, a in enumerate(
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
            ):
                ax[a].set_xticks([])
                ax[a].set_yticks([])
                if len(self.times) > 1:
                    video = self.iarr[frames[i]]
                    ax[a].imshow(
                        video,
                        extent=extent,
                        vmin=vmin,
                        vmax=f_vmax,
                        cmap='inferno'
                    )
                    ax[a].set_title(
                        f'{self.times[frames[i]]:.2f} UT',
                        fontsize=14
                    )
                    ax[a].text(
                        0.98,
                        0.98,
                        f'#{frames[i]}',
                        c='w',
                        ha='right',
                        va='top',
                        fontsize=14,
                        transform=ax[a].transAxes
                    )
                    ax[a].text(
                        0.98,
                        0.02,
                        f'$S_{{tot}}$: {self.iarr[frames[i]].sum():.1f} Jy',
                        c='w',
                        ha='right',
                        va='bottom',
                        fontsize=12,
                        transform=ax[a].transAxes
                    )
                else:
                    video = np.ones((self.npix, self.npix)) * np.nan
                    ax[a].imshow(video, extent=extent)
            # Labeling
            if len(self.times) > 1:
                ax['l'].text(
                    0.1,
                    -0.04,
                    f'{round(barl):1d} $\mu$as',
                    c='k',
                    ha='center',
                    va='top',
                    transform=ax['l'].transAxes
                )
                ax['l'].hlines(
                    -0.03,
                    0,
                    0.2,
                    transform=ax['l'].transAxes,
                    colors='k',
                    lw=3,
                    clip_on=False
                )
            # Dynamic component frames
            ax['m'].text(
                0,
                1.15,
                'Dynamic component',
                c='k',
                weight='bold',
                fontsize=16,
                transform=ax['m'].transAxes
            )
            for i, a in enumerate(
                ['m', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w']
            ):
                ax[a].set_xticks([])
                ax[a].set_yticks([])
                if len(self.times) > 1:
                    dynamic = d_out[frames[i], ..., 0].reshape(
                        self.npix, self.npix
                    )
                    ax[a].imshow(
                        dynamic,
                        extent=extent,
                        cmap='magma_r',
                        vmin=vmin,
                        vmax=d_vmax
                    )
                    ax[a].contour(
                        s_out[..., 0].reshape(self.npix, self.npix),
                        extent=extent,
                        levels=3,
                        colors='k',
                        origin='upper',
                        alpha=0.5
                    )
                    ax[a].set_title(
                        f'{self.times[frames[i]]:.2f} UT',
                        fontsize=14
                    )
                    ax[a].text(
                        0.98,
                        0.98,
                        f'#{frames[i]}',
                        c='k',
                        ha='right',
                        va='top',
                        fontsize=14,
                        transform=ax[a].transAxes
                    )
                    ax[a].text(
                        0.98,
                        0.02,
                        f'$S_{{tot}}$: {dynamic.sum():.1f} Jy',
                        c='k',
                        ha='right',
                        va='bottom',
                        fontsize=12,
                        transform=ax[a].transAxes
                    )
                else:
                    dynamic = np.ones((self.npix, self.npix)) * np.nan
                    ax[a].imshow(dynamic, extent=extent)
            # Static component
            static = s_out[..., 0].reshape(self.npix, self.npix)
            ax['x'].imshow(static, extent=extent, cmap='RdPu')
            ax['x'].set_title('Static component', weight='bold', fontsize=16)
            ax['x'].set_xticks([])
            ax['x'].set_yticks([])
            ax['x'].text(
                0.98,
                0.02,
                f'$S_{{tot}}$: {static.sum():.1f} Jy',
                c='k',
                ha='right',
                va='bottom',
                fontsize=12,
                transform=ax['x'].transAxes
            )
            ax['x'].text(
                0.1,
                -0.04,
                f'{round(barl):1d} $\mu$as',
                c='k',
                ha='center',
                va='top',
                transform=ax['x'].transAxes
            )
            ax['x'].hlines(
                -0.03,
                0,
                0.2,
                transform=ax['x'].transAxes,
                colors='k',
                lw=3,
                clip_on=False
            )
            # Amplitude gains
            axz = ax['s'].inset_axes([0, -2.3, 2.8, 2])
            axz.text(0, 1.03, 'Vis. amp. gains', c='k', weight='bold',
                    fontsize=16, transform=axz.transAxes)
            # Colors
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            # Plotting
            if self.amp_gains is not None:
                for i in range(len(self.amp_gains['sites'])-1):
                    axz.plot(
                        self.times,
                        self.amp_gains['gains'][i, :] + i/2,
                        'o',
                        ms=7,
                        lw=1,
                        c=colors[i],
                        label=f"{self.amp_gains['sites'][i]}"
                    )
                    axz.plot(
                        self.times,
                        self.amp_gains['gains'][i, :] + i/2,
                        c=colors[i]
                    )
                    axz.axhline(1+i/2, c='gray', lw=1, zorder=0)
                axz.set_ylim(0.5, len(self.amp_gains['sites'])/2+0.5)
                axz.set_yticks([0.5, 1, 1.5])
                axz.set_xlabel('Time (UT)')
                axz.legend(
                    loc='upper right',
                    ncols=len(self.amp_gains['sites'])-1,
                    bbox_to_anchor=(1.01, 1.08),
                    columnspacing=1,
                    handletextpad=0.1,
                    markerscale=1.5
                )
            else:
                axz.set_xticks([])
                axz.set_yticks([])
            # Loss
            axz = ax['x'].inset_axes([-2, -2.3, 3, 2])
            axz.text(
                0,
                1.03,
                'Loss',
                c='k',
                weight='bold',
                fontsize=16,
                transform=axz.transAxes
            )
            if isinstance(self.loss, dict):
                for l in self.loss:
                    axz.plot(
                        np.linspace(
                            0,
                            len(self.loss[l]),
                            len(self.loss[l][::50])
                        ),
                        self.loss[l][::50],
                        label=f'{l} = {self.loss[l][-1]:.3e}'
                    )
                    axz.axhline(1, c='k', lw=0.5, zorder=0)
            else:
                axz.plot(
                    np.linspace(
                        0,
                        len(self.loss),
                        len(self.loss[::50])
                    ),
                    self.loss[::50],
                    label=f'loss = {self.loss[-1]:.3e}'
                )
            axz.set_yscale('log')
            axz.set_xlim(-self.niter/50, self.niter+self.niter/50)
            axz.set_xlabel('iterations')
            axz.legend(
                loc='upper right',
                bbox_to_anchor=(1, 1),
                fontsize=12,
                ncols=len(self.loss)//2
            )
            axz.grid(alpha=0.5)
            # Save and close
            fig.subplots_adjust(wspace=0.01)

        # Show interactive plot or save to file
        if show:
            plt.show()
        else:
            fig.savefig(outpath, bbox_inches='tight')
            plt.close()

    def plot_gif(
            self,
            s_out: ArrayLike | None = None,
            d_out: ArrayLike | None = None,
            scale: str = 'lin',
            drange: float = 1e3,
            vstep: int = 2,
            vscale: float = 0.05,
            outpath='./video.gif'
    ) -> None:
        """Plot gif from Video object in full pol.

        Args:
            s_out: Static component video array.
            d_out: Dynamic component video array.
            scale: 'lin' for linear color scale, 'log' for logarithmic.
            drange: Image dynamic range. Used when scale='log'.
            vstep: Pol. field vectors' spacing.
            vscale: Pol. field vectors' scaling factor.
            outpath: Path and filename for figure saving.
        """
        # Preamble
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 14
        plt.rcParams['image.interpolation'] = 'bicubic'
        # Extent
        extent = [
            self.fov/2/eh.RADPERUAS,
            -self.fov/2/eh.RADPERUAS,
            -self.fov/2/eh.RADPERUAS,
            self.fov/2/eh.RADPERUAS
        ]
        # Scale bar
        barl = self.fov / eh.RADPERUAS / 5
        # Set colorbar scale
        def normalize(arr, scale=scale, drange=drange):
            vmin, vmax = arr.min(), arr.max()
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            if scale == 'log':
                vmin = vmax / drange
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            return vmin, vmax, norm
        # Number of axes
        video = {'video': self.iarr}
        if s_out is not None:
            video |= {
                'static': s_out[..., 0].reshape(self.npix, self.npix)
            }
        if d_out is not None:
            video |= {
                'dynamic': d_out[..., 0].reshape(-1, self.npix, self.npix)
            }
        # Init figure
        fig, ax = plt.subplots(1, len(video))
        fig.set_size_inches(5*len(video), 5.7)
        fig.subplots_adjust(
            left=0.005,
            right=0.995,
            bottom=0.02,
            top=1,
            wspace=0.01)
        ax = np.atleast_1d(ax)
        # Pop static from dict
        s_out = video.pop('static', None)
        if s_out is not None:
            ax[-1].imshow(s_out, extent=extent, cmap='RdPu')
            ax[-1].set_title('Static component')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].text(
                1,
                -0.01,
                f'$S_{{tot}}$: {s_out.sum():.1f} Jy',
                c='k',
                ha='right',
                va='top',
                fontsize=10,
                transform=ax[-1].transAxes
            )
        # Plot frames iteratively
        def plot_frame(f):
            for i, key in enumerate(video):
                cmap = 'magma_r' if key == 'dynamic' else 'inferno'
                ax[i].clear()
                vmin, _, norm = normalize(video[key])
                ax[i].imshow(
                    video[key][f] + vmin,
                    extent=extent,
                    cmap=cmap,
                    norm=norm
                )
                if i == 0 and 'dynamic' in video:
                    ax[i].set_title('Full video')
                if i == 1:
                    ax[i].set_title('Dynamic component')
                c = 'k' if key == 'dynamic' else 'w'
                if not isinstance(self.dates, list):
                    if 'dynamic' in video:
                        ax[i].set_xlabel(
                            f'{self.times[f]:.2f} UT',
                            labelpad=-345,
                            c=c
                        )
                    else:
                        ax[i].set_title(
                            f'{self.times[f]:.2f} UT',
                            c='k',
                            fontsize=14
                        )
                else:
                    if 'dynamic' in video:
                        ax[i].set_xlabel(
                            f'{self.dates[f]}',
                            labelpad=-345,
                            c=c
                        )
                    else:
                        ax[i].set_title(
                            f'{self.dates[f]}',
                            c='k',
                            fontsize=14
                        )
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].text(
                    1,
                    -0.01,
                    f'$S_{{tot}}$: {video[key][f].sum():.1f} Jy',
                    c='k',
                    ha='right',
                    va='top',
                    fontsize=10,
                    transform=ax[i].transAxes
                )
                if i == 1:
                    ax[i].contour(
                        s_out,
                        extent=extent,
                        levels=3,
                        colors='k',
                        origin='upper',
                        alpha=0.5
                    )
                # Polarization
                if self.qarr is not None and i == 0:
                    x = np.linspace(extent[0], extent[1], self.npix)
                    y = np.linspace(extent[0], extent[1], self.npix)
                    vx = np.sin(
                        np.angle(self.qarr[f] + 1j * self.uarr[f]) / 2
                    ) * self.parr[f] * -1
                    vy = np.cos(
                        np.angle(self.qarr[f] + 1j * self.uarr[f]) / 2
                    ) * self.parr[f]
                    ax[i].quiver(
                        x[::vstep],
                        y[::vstep],
                        vx[::vstep, ::vstep, 0],
                        vy[::vstep, ::vstep, 0],
                        self.larr[f, ::vstep, ::vstep, 0],
                        pivot='mid',
                        angles='uv',
                        width=0.006,
                        scale=vscale,
                        headwidth=0,
                        headlength=0,
                        headaxislength=0,
                        cmap='viridis'
                    )
            # Scale bar
            ax[0].text(
                0.1,
                -0.03,
                f'{round(barl):1d} $\mu$as',
                c='k',
                ha='center',
                va='top',
                transform=ax[0].transAxes
            )
            ax[0].hlines(
                -0.02,
                0,
                0.2,
                transform=ax[0].transAxes,
                colors='k',
                lw=3,
                clip_on=False
            )
            return fig
        # Helper function
        def update(f):
            return plot_frame(f)
        # Define writer
        fps = 10
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(len(self.times)),
            interval=1e3/fps
        )
        wri = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)
        # Save gif
        ani.save(outpath, writer=wri, dpi=100)

    @staticmethod
    def async_plot(q: Callable) -> None:
        """Asynchronous plotting routine.

        Plot results from CPU without stopping GPU computations.
        It loads the network output on a separate thread.

        Note:
            Not very matplotlib-safe because of threading, but it
            works just fine so far. Warnings may appear occasionally.
        
        Args:
            q: Queue object where output is loaded.
        
        Todo:
            * Make sure arrays are transferred to CPU.
        """
        def _async_plot_impl(
                *,
                video,
                out,
                **kwargs,
        ):
            video.from_video(out)
            video.loss = kwargs.pop('loss', None)
            video.amp_gains = kwargs.pop('amp_gains', None)
            video.plot(**kwargs)
        # Unpack queue
        while True:
            _async_plot_impl(**q.get())
            q.task_done()

    def save_gains(self, outpath: str = './gains.txt') -> None:
        """Save fitted gains to text file.

        Args:
            outpath: Path and filename of output file.
        """
        with open(outpath, 'w') as f:
            f.write('# UT (h) ')
            for site in self.amp_gains['sites']:
                f.write(f'{site} ')
            f.write('\n')
            for i, _ in enumerate(self.times):
                f.write(f'{self.times[i]} ')
                for j in range(len(self.amp_gains['sites'])):
                    f.write(f"{self.amp_gains['gains'][j, i]} ")
                f.write('\n')

    def save_fits(self, outpath: str = './image.fits') -> None:
        """Save image to fits file.

        Args:
            outpath: Path and filename of output file.
        """
        image = eh.image.make_empty(self.npix, self.fov, self.ra, self.dec)
        image.imvec = self.iarr[0, ..., 0].ravel()
        if self.qarr is not None:
            image.add_qu(self.qarr[0, ..., 0], self.uarr[0, ..., 0])
        if self.varr is not None:
            image.add_v(self.varr[0, ..., 0])
        with ut.no_print():
            image.save_fits(outpath)

    def save_h5(self, outpath: str = './video.h5') -> None:
        """Save video to h5 file.

        Args:
            outpath: Path and filename of output file.
        
        Todo:
            * Fix mjd
        """
        frames = []
        proxy = eh.image.make_empty(self.npix, self.fov, self.ra, self.dec)
        for i, _ in enumerate(self.times):
            frame = proxy.copy()
            frame.time = self.times.tolist()[i]
            frame.imvec = self.iarr[i, ..., 0].ravel()
            if self.qarr is not None:
                frame.add_qu(self.qarr[i, ..., 0], self.uarr[i, ..., 0])
            if self.varr is not None:
                frame.add_v(self.varr[i, ..., 0])
            frames.append(frame)
        with ut.no_print():
            eh.movie.merge_im_list(frames).save_hdf5(outpath)
