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

"""Trainer object. A compilation of useful methods for training."""

from typing import Any
from collections import OrderedDict

import numpy as np
from flax.training import train_state
import jax
from jax import Array
from jax import numpy as jnp
try:
    from jax_finufft import nufft2
except ImportError:
    print('Warning: NUFFT not installed \n')

from . import utils as ut


NPIX: int = 0
"""Global variable required for NUFFT computations.

This is currently the best way I've found for passing this value
to a @jax.jit function. NPIX should be updated from main. See examples.
"""


class Trainer(train_state.TrainState):
    """Compilation of useful functions for training.

    Adds batch normalization to flax.training.train_state.TrainState.
    In addition, all functions related to training, like loss functions,
    are included here as (mostly private) static methods for consistency.

    Args:
        batch_stats: Batch normalization statistics
    """

    batch_stats: Any | None = None

    @staticmethod
    def _which_loss_fn(*args, **kwargs):
        """Select the loss function based on the input arguments."""
        if 'min_lcurve' not in kwargs:
            if 'grid' in kwargs and kwargs['grid'].shape[-1] < 3:
                if 'init_img' in kwargs:
                    return Trainer._loss_fn_init_2d(*args, **kwargs)
                return Trainer._loss_fn_2d(*args, **kwargs)
            if 'init_vid' in kwargs:
                return Trainer._loss_fn_init(*args, **kwargs)
            if 'init_vid_ml' in kwargs:
                return Trainer._loss_fn_init_pol(*args, **kwargs)
            if 'uvpoints' in kwargs:
                return Trainer._loss_fn_nfft(*args, **kwargs)
            if 'init_vid_i' in kwargs:
                return Trainer._loss_fn_pol(*args, **kwargs)
            if 's_grid' in kwargs:
                return Trainer._loss_fn_reg_gains(*args, **kwargs)
            return Trainer._loss_fn(*args, **kwargs)
        if 'uvpoints' in kwargs:
            return Trainer._loss_fn_div_nfft(*args, **kwargs)
        return Trainer._loss_fn_div_gains(*args, **kwargs)

    @staticmethod
    def _loss_fn(*args, **kwargs):
        """Loss function for training a general dynamic network."""
        # Unpack states
        params, batch_stats, apply_fn = args
        # Unpack kwargs
        data = kwargs.get('data')
        grid = kwargs.get('grid')
        lcurve = kwargs.get('lcurve')
        # Get video and batch stats
        video, updates = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            grid,
            train=True,
            mutable=['batch_stats']
        )
        # Separate polarimetric channels
        polchan = ['I']
        if video.shape[-1] == 4:
            polchan = ['I', 'Q', 'U']
        if video.shape[-1] == 5:
            polchan = ['I', 'Q', 'U', 'V']
        videopol = {
            pol: ut.to_complex(video[..., 0]) if pol == 'I'
            else ut.to_complex(
                -video[..., 0] * video[..., 1] * video[..., 2]
            ) if pol == 'Q'
            else ut.to_complex(
                video[..., 0] * video[..., 1] * video[..., 3]
            ) if pol == 'U'
            else ut.to_complex(
                video[..., 0] * video[..., 4]
            )
            for pol in polchan
        }
        # Initialize loss dictionary
        ldict = {}
        # Data product loss functions
        for dtype in data:
            pol = dtype[-1]
            chi2 = Trainer._loss_chi(
                videopol[pol],
                data[dtype],
                dtype[:-1]
            )
            ldict[dtype] = chi2
        # Regularizer loss functions
        if lcurve is not None:
            chi2 = Trainer._loss_lcurve(
                lcurve,
                videopol['I']
            )
            ldict['lcurve'] = chi2
        # Total loss
        loss = jnp.sum(jnp.array(list(ldict.values())))
        return loss, (updates, ldict, (video,))

    @staticmethod
    def _loss_fn_nfft(*args, **kwargs):
        """Loss function for training a general dynamic network using NUFFT."""
        # Unpack states
        params, batch_stats, apply_fn = args
        # Unpack kwargs
        data = kwargs.get('data')
        grid = kwargs.get('grid')
        lcurve = kwargs.get('lcurve')
        uvpoints = kwargs.get('uvpoints')
        pulses = kwargs.get('pulsefac')
        uvind = kwargs.get('uvind')
        triangles = kwargs.get('triangles')
        quadrangles = kwargs.get('quadrangles')
        # Get video and batch stats
        video, updates = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            grid, train=True, mutable=['batch_stats']
            )
        # Separate polarimetric channels
        videopol = {'I': ut.to_complex(video[...,0])}
        # Initialize loss dictionary
        ldict = {}
        # Data product loss functions
        for dtype in data:
            chi2 = Trainer._loss_chi(
                videopol['I'],
                data[dtype],
                dtype[:-1],
                uv=uvpoints,
                pulses=pulses,
                uvind=uvind,
                tria=triangles,
                quad=quadrangles,
            )
            ldict[dtype] = chi2
        # Regularizer loss functions
        chi2 = Trainer._loss_lcurve(
            lcurve,
            videopol['I']
            )
        ldict['lcurve'] = chi2
        # Total loss
        loss = jnp.sum(jnp.array(list(ldict.values())))
        return loss, (updates, ldict, (video,))

    @staticmethod
    def _loss_fn_pol(*args, **kwargs):
        """Loss function for training a general dynamic network
        for Stokes Q and U having Stokes I fixed."""
        # Unpack states
        params, batch_stats, apply_fn = args
        # Unpack kwargs
        data = kwargs.get('data')
        grid = kwargs.get('grid')
        init_vid_i = kwargs.get('init_vid_i')
        init_vid_i = init_vid_i.reshape(init_vid_i.shape[0], -1)
        # Get video and batch stats
        video, updates = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            grid, train=True, mutable=['batch_stats']
            )
        # Initialize loss dictionary
        ldict = {}
        # Make ml live near stokes I
        chi2 = Trainer._loss_ml_overlap(
            video[..., 0],
            init_vid_i,
            tau = 0.01
            )
        ldict['overlap'] = chi2
        # Separate polarimetric channels
        videopol = {
            'I': ut.to_complex(init_vid_i),
            'Q': ut.to_complex(-init_vid_i * video[..., 0] * video[..., 1]),
            'U': ut.to_complex(init_vid_i * video[..., 0] * video[..., 2])
            }
        # Data product loss functions
        for dtype in data:
            pol = dtype[-1]
            chi2 = Trainer._loss_chi(
                videopol[pol],
                data[dtype],
                dtype[:-1],
                )
            ldict[dtype] = chi2
        # Total loss
        loss = jnp.sum(jnp.array(list(ldict.values())))
        return loss, (updates, ldict, (video,))

    @staticmethod
    def _loss_fn_2d(*args, **kwargs):
        """Loss function for training a general static network."""
        # Unpack state
        params, batch_stats, apply_fn = args
        # Unpack kwargs
        data = kwargs.get('data')
        coords = kwargs.get('coords')
        lcurve = kwargs.get('lcurve')
        # Get video and batch stats
        image, updates = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            coords, train=True, mutable=['batch_stats']
            )
        # Separate polarimetric channels
        polchan = ['I']
        if image.shape[-1] == 4:
            polchan = ['I', 'Q', 'U']
        if image.shape[-1] == 5:
            polchan = ['I', 'Q', 'U', 'V']
        imagepol = {
            pol: ut.to_complex(image[..., 0]) if pol == 'I'
            else ut.to_complex(
                -image[..., 0] * image[..., 1] * image[..., 2]
                               ) if pol == 'Q'
            else ut.to_complex(
                image[..., 0] * image[..., 1] * image[..., 3]
                               ) if pol == 'U'
            else ut.to_complex(
                image[..., 0] * image[..., 4]
                               )
            for pol in polchan
        }
        # Initialize loss dictionary
        ldict = {}
        # Data product loss functions
        for dtype in data:
            pol = dtype[-1]
            chi2 = Trainer._loss_chi(
                imagepol[pol],
                data[dtype],
                dtype[:-1]
                )
            ldict[dtype] = chi2
        # Regularizer loss functions
        if lcurve is not None:
            chi2 = Trainer._loss_lcurve_2d(
                lcurve,
                imagepol['I']
                )
            ldict['lcurve'] = chi2
        # Total loss
        loss = jnp.sum(jnp.array(list(ldict.values())))
        return loss, (updates, ldict, (image,))

    @staticmethod
    def _loss_gains(
        data,
        ag_apply_fn,
        ag_params,
        pg_apply_fn,
        pg_params,
        bl_indx
    ):
        """Select which gain correction to apply based onthe data products
        used for imaging."""
        if any(x in data for x in ['ampI', 'logampI']):
            dtype = 'ampI' if 'ampI' in data else 'logampI'
            frames = jnp.arange(len(data[dtype]['target']), dtype=int)
            visamp = data[dtype]['target'].copy()
            agi, agj = ag_apply_fn({'params': ag_params}, bl_indx, frames)
            return agi * agj * visamp

        if 'visI' in data:
            compvis = data['visI']['target'].copy()
            frames = jnp.arange(len(data['visI']['target']), dtype=int)
            agi, agj = ag_apply_fn({'params': ag_params}, bl_indx, frames)
            pgi, pgj = pg_apply_fn({'params': pg_params}, bl_indx, frames)
            gi, gj = agi * jnp.exp(1j * pgi), agj * jnp.exp(1j * pgj)
            return gi * jnp.conj(gj) * compvis

    @staticmethod
    def _loss_fn_div_gains(*args, **kwargs):
        """Loss function for training a static and dynamic network
        with gain corrections."""
        # Unpack states
        s_params, d_params, ag_params, pg_params, \
        s_batch_stats, d_batch_stats, s_apply_fn, \
        d_apply_fn, ag_apply_fn, pg_apply_fn = args
        # Unpack kwargs
        data = kwargs.get('data')
        s_grid = kwargs.get('s_grid')
        d_grid = kwargs.get('d_grid')
        lcurve = kwargs.get('lcurve')
        min_lcurve = kwargs.get('min_lcurve')
        bl_indx = kwargs.get('bl_indx')
        w_border = kwargs.get('w_border', 1e3)
        w_flux = kwargs.get('w_flux', 1e3)
        # Get static and dynamic videos and batch stats
        static, s_updates  = s_apply_fn(
            {'params': s_params, 'batch_stats': s_batch_stats},
            s_grid, train=True, mutable=['batch_stats']
            )
        dynamic, d_updates = d_apply_fn(
            {'params': d_params, 'batch_stats': d_batch_stats},
            d_grid, train=True, mutable=['batch_stats']
            )
        # Initialize loss dictionary
        ldict = {}
        # Border regularization
        loss_bd = w_border * Trainer._loss_border(static[..., 0])
        loss_bd += w_border * Trainer._loss_border(dynamic[..., 0])
        ldict['border'] = loss_bd
        # Flux regularization
        # Enforce a total flux of 1 in both parts
        loss_flux_sta = w_flux * Trainer._loss_static_flux(static[..., 0])
        loss_flux_dyn = w_flux * Trainer._loss_dynamic_flux(dynamic[..., 0])
        ldict['s_flux'] = loss_flux_sta
        ldict['d_flux'] = loss_flux_dyn
        # Now rescale accordingly and add the static and dynamic outputs
        modlcurve = lcurve - min_lcurve
        modlcurve = modlcurve.reshape(-1, 1, 1)
        static *= min_lcurve
        dynamic *= modlcurve
        video = static + dynamic
        # Separate polarimetric channels
        videopol = {'I': ut.to_complex(video[...,0])}
        # Apply gain corrections
        gain_corr_data = Trainer._loss_gains(
            data,
            ag_apply_fn,
            ag_params,
            pg_apply_fn,
            pg_params,
            bl_indx
            )
        # Data product loss functions
        for dtype in data:
            chi2 = Trainer._loss_chi(
                videopol['I'],
                data[dtype],
                dtype[:-1],
                gain_corr_data=gain_corr_data
                )
            ldict[dtype] = chi2
        # Total loss
        loss = jnp.sum(jnp.array(list(ldict.values())))
        return loss, (s_updates, d_updates, ldict, (static, dynamic, video))

    @staticmethod
    def _loss_fn_div_gains_flux(*args, **kwargs):
        """
        Loss function for training a static and dynamic network
        with gain corrections and learnable flux ration."""
        # Unpack states
        s_params, d_params, fd_params, ag_params, pg_params, \
            s_batch_stats, d_batch_stats, s_apply_fn, d_apply_fn, \
                fd_apply_fn, ag_apply_fn, pg_apply_fn = args
        # Unpack kwargs
        data = kwargs.get('data')
        s_grid = kwargs.get('s_grid')
        d_grid = kwargs.get('d_grid')
        lcurve = kwargs.get('lcurve')
        min_lcurve = kwargs.get('min_lcurve')
        bl_indx = kwargs.get('bl_indx')
        w_border = kwargs.get('w_border', 1e3)
        w_flux = kwargs.get('w_flux', 1e3)
        # Get static and dynamic videos and batch stats
        static, s_updates  = s_apply_fn(
            {'params': s_params, 'batch_stats': s_batch_stats},
            s_grid, train=True, mutable=['batch_stats']
            )
        dynamic, d_updates = d_apply_fn(
            {'params': d_params, 'batch_stats': d_batch_stats},
            d_grid, train=True, mutable=['batch_stats']
            )
        # Initialize loss dictionary
        ldict = {}
        # Border regularization
        loss_bd = w_border * Trainer._loss_border(static[..., 0])
        loss_bd += w_border * Trainer._loss_border(dynamic[..., 0])
        ldict['border'] = loss_bd
        # Flux regularization
        # Enforce a total flux of 1 in both parts
        loss_flux_sta = w_flux * Trainer._loss_static_flux(static[..., 0])
        loss_flux_dyn = w_flux * Trainer._loss_dynamic_flux(dynamic[..., 0])
        ldict['s_flux'] = loss_flux_sta
        ldict['d_flux'] = loss_flux_dyn
        # Find static flux density
        min_lcurve = fd_apply_fn({'params': fd_params})
        # Now rescale accordingly and add the static and dynamic outputs
        modlcurve = lcurve - min_lcurve
        modlcurve = modlcurve.reshape(-1, 1, 1)
        video = static * min_lcurve + dynamic * modlcurve
        # Separate polarimetric channels
        videopol = {'I': ut.to_complex(video[...,0])}
        # Apply gain corrections
        gain_corr_data = Trainer._loss_gains(
            data,
            ag_apply_fn,
            ag_params,
            pg_apply_fn,
            pg_params,
            bl_indx
            )
        # Data product loss functions
        for dtype in data:
            chi2 = Trainer._loss_chi(
                videopol['I'],
                data[dtype],
                dtype[:-1],
                gain_corr_data=gain_corr_data
                )
            ldict[dtype] = chi2
        # Total loss
        loss = jnp.sum(jnp.array(list(ldict.values())))
        return loss, \
            (s_updates, d_updates, ldict, (static, dynamic, video, min_lcurve))
    
    @staticmethod
    def _loss_fn_reg_gains(*args, **kwargs):
        """Loss function for training a static and dynamic network
        while finding the static flux density through regularization."""
        # Unpack states
        s_params, d_params, ag_params, pg_params, \
            s_batch_stats, d_batch_stats, s_apply_fn, \
                d_apply_fn, ag_apply_fn, pg_apply_fn = args
        # Unpack kwargs
        data = kwargs.get('data')
        s_grid = kwargs.get('s_grid')
        d_grid = kwargs.get('d_grid')
        bl_indx = kwargs.get('bl_indx')
        w_border = kwargs.get('w_border', 1e3)
        w_flux = kwargs.get('w_flux', 5)
        # Get static and dynamic videos and batch stats
        static, s_updates  = s_apply_fn(
            {'params': s_params, 'batch_stats': s_batch_stats},
            s_grid, train=True, mutable=['batch_stats']
            )
        dynamic, d_updates = d_apply_fn(
            {'params': d_params, 'batch_stats': d_batch_stats},
            d_grid, train=True, mutable=['batch_stats']
            )
        # Initialize loss dictionary
        ldict = {}
        # Border regularization
        loss_bd = w_border * Trainer._loss_border(static[..., 0])
        loss_bd += w_border * Trainer._loss_border(dynamic[..., 0])
        ldict['border'] = loss_bd
        # Flux regularization
        # Minimize persistent flux in dynamic component
        loss_min_dyn = w_flux * Trainer._loss_min_dynamics(dynamic[..., 0])
        ldict['min_dyn'] = loss_min_dyn
        # Add the static and dynamic outputs
        video = static + dynamic
        # Separate polarimetric channels
        videopol = {'I': ut.to_complex(video[...,0])}
        # Apply gain corrections
        gain_corr_data = Trainer._loss_gains(
            data,
            ag_apply_fn,
            ag_params,
            pg_apply_fn,
            pg_params,
            bl_indx
            )
        # Data product loss functions
        for dtype in data:
            chi2 = Trainer._loss_chi(
                videopol['I'],
                data[dtype],
                dtype[:-1],
                gain_corr_data=gain_corr_data
                )
            ldict[dtype] = chi2
        # Total loss
        loss = jnp.sum(jnp.array(list(ldict.values())))
        return loss, (s_updates, d_updates, ldict, (static, dynamic, video))

    @staticmethod
    def _loss_fn_init(*args, **kwargs):
        """Loss function for initializing a general dynamic network."""
        # Unpack states
        params, batch_stats, apply_fn = args
        # Unpack kwargs
        grid = kwargs.get('grid')
        init_vid = kwargs.get('init_vid')
        # Get video and batch stats
        video, updates = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            grid, train=True, mutable=['batch_stats']
            )
        # Separate polarimetric channels
        polchan = ['I']
        if video.shape[-1] == 4:
            polchan = ['I', 'Ml', 'X']
        if video.shape[-1] == 5:
            polchan = ['I', 'Ml', 'X', 'Mc']
        videopol = {
            pol: video[..., 0] if pol == 'I'
            else video[..., 1] if pol == 'Ml'
            else jnp.arctan2(video[..., 2], video[..., 3]) * 0.5 if pol == 'X'
            else video[..., 4]
            for pol in polchan
            }
        initpol = {pol: init_vid[..., i] for i, pol in enumerate(polchan)}
        # Initialize loss
        loss = []
        # Loss function for data products
        for pol in polchan:
            chi2 = jnp.mean(jnp.square(videopol[pol]
                 - initpol[pol].reshape(videopol[pol].shape[0], -1))
                 )
            loss.append(chi2)
        return jnp.sum(jnp.array(loss)), (updates, None, (video,))

    @staticmethod
    def _loss_fn_init_pol(*args, **kwargs):
        """Loss function for initializing a general dynamic network
        for Stokes Q and U having Stokes I fixed."""
        # Unpack states
        params, batch_stats, apply_fn = args
        # Unpack kwargs
        grid = kwargs.get('grid')
        init_vid_ml = kwargs.get('init_vid_ml')
        init_vid_x = kwargs.get('init_vid_x')
        # Get video and batch stats
        video, updates = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            grid, train=True, mutable=['batch_stats']
            )
        # Separate polarimetric channels
        polchan = ['Ml', 'X']
        videopol = {
            pol: video[..., 0] if pol == 'Ml'
            else jnp.arctan2(video[..., 1], video[..., 2]) * 0.5
            for _, pol in enumerate(polchan)
            }
        initpol = {
            pol: init_vid_ml if pol == 'Ml'
            else init_vid_x
            for _, pol in enumerate(polchan)
            }
        # Initialize loss dictionary
        loss = []
        # Loss function for data products
        for pol in polchan:
            chi2 = jnp.mean(jnp.square(videopol[pol]
                 - initpol[pol].reshape(videopol[pol].shape[0], -1))
                 )
            loss.append(chi2)
        return jnp.sum(jnp.array(loss)), (updates, None, (video,))

    @staticmethod
    def _loss_fn_init_2d(*args, **kwargs):
        """Loss function for initializing a general static network."""
        # Unpack state
        params, batch_stats, apply_fn = args
        # Unpack kwargs
        grid = kwargs.get('grid')
        init_img = kwargs.get('init_img')
        # Get video and batch stats
        image, updates = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            grid, train=True, mutable=['batch_stats']
            )
        # Separate polarimetric channels
        polchan = ['I']
        if image.shape[-1] == 4:
            polchan = ['I', 'Ml', 'X']
        if image.shape[-1] == 5:
            polchan = ['I', 'Ml', 'X', 'Mc']
        imagepol = {
            pol: image[..., 0] if pol == 'I'
            else image[..., 1] if pol == 'Ml'
            else jnp.arctan2(image[..., 2], image[..., 3]) * 0.5 if pol == 'X'
            else image[..., 4]
            for pol in polchan
            }
        initpol = {pol: init_img[..., i] for i, pol in enumerate(polchan)}
        # Initialize loss
        loss = []
        # Loss function for data products
        for pol in polchan:
            chi2 = jnp.mean(jnp.square(imagepol[pol]
                 - initpol[pol].flatten())
                 )
            loss.append(chi2)
        return jnp.sum(jnp.array(loss)), (updates, None, (image,))

    @staticmethod
    def _loss_chi(video, data, dtype, **kwargs):
        """Compute loss for a given data product."""
        # Check if static (no padmask) or dynamic imaging
        if 'padmask' in data:
            # Check if using NUFFT
            if 'uv' in kwargs:
                # Unpack NUFTT variables
                uv = kwargs['uv']
                pulses = kwargs['pulses']
                uvind = kwargs['uvind']
                tria = kwargs['tria']
                quad = kwargs['quad']
                # Parallelize NUFFT with jax.vmap
                # (eps=1e-3 works just fine)
                def single_nufft(img, v, u, pulse):
                    return nufft2(img, v, u, eps=1e-3) * pulse
                multi_nufft = jax.vmap(single_nufft, in_axes=(0, 0, 0, 0))
                # Get visibilites from the NUFFT
                batch = jnp.arange(video.shape[0])[:, None]
                video = video.reshape(video.shape[0], NPIX, NPIX)
                vis = multi_nufft(video, uv['v'], uv['u'], pulses)
                # Select loss based on dtype
                if dtype == 'vis':
                    return Trainer._loss_vis_nfft(vis[batch, uvind], data)
                if dtype == 'amp':
                    amp  = jnp.abs(vis[batch, uvind])
                    return Trainer._loss_amp_nfft(amp[batch, uvind], data)
                if dtype == 'logamp':
                    eps = 1e-12
                    logamp = jnp.abs(vis[batch, uvind])
                    logamp = jnp.log(logamp + eps)
                    return Trainer._loss_logamp_nfft(logamp, data)
                if dtype == 'cphase':
                    expvis = vis[:, None, :, None]
                    exptria = tria[..., None]
                    whichvis = jnp.take_along_axis(
                        expvis,
                        exptria,
                        axis=2
                    ).squeeze(-1)
                    cphase = jnp.prod(whichvis, axis=2)
                    cphase = jnp.angle(cphase)
                    return Trainer._loss_cphase_nfft(cphase, data)
                if dtype == 'logcamp':
                    eps = 1e-12
                    expvis = vis[:, None, :, None]
                    expquad = quad[..., None]
                    whichvis = jnp.take_along_axis(
                        expvis,
                        expquad,
                        axis=2
                    ).squeeze(-1)
                    whichvis = jnp.log(jnp.abs(whichvis) + eps)
                    logcamp = whichvis[..., 0] \
                            + whichvis[..., 1] \
                            - whichvis[..., 2] \
                            - whichvis[..., 3]
                    return Trainer._loss_logcamp_nfft(logcamp, data)
            else:
                # Unpack gains
                gain_corr_data = kwargs.get('gain_corr_data', None)
                # Select loss based on dtype
                if dtype == 'vis':
                    return Trainer._loss_vis(video, data, gain_corr_data)
                if dtype == 'amp':
                    return Trainer._loss_amp(video, data, gain_corr_data)
                if dtype == 'logamp':
                    return Trainer._loss_logamp(video, data, gain_corr_data)
                if dtype == 'cphase':
                    return Trainer._loss_cphase(video, data)
                if dtype == 'logcamp':
                    return Trainer._loss_logcamp(video, data)
                if dtype == 'bs':
                    return Trainer._loss_bs(video, data)
                if dtype == 'mbreve':
                    return Trainer._loss_mbreve(video, data)
        else:
            # Select loss based on dtype
            if dtype == 'vis':
                return Trainer._loss_vis_2d(video, data)
            if dtype == 'amp':
                return Trainer._loss_amp_2d(video, data)
            if dtype == 'logamp':
                return Trainer._loss_logamp_2d(video, data)
            if dtype == 'cphase':
                return Trainer._loss_cphase_2d(video, data)
            if dtype == 'logcamp':
                return Trainer._loss_logcamp_2d(video, data)
            if dtype == 'bs':
                return Trainer._loss_bs_2d(video, data)

    @staticmethod
    def _loss_vis(video, data, gvis):
        gvis = data['target'] if gvis is None else gvis
        vis = jax.lax.batch_matmul(
            data['A'][:, 0, ...], video
        ).squeeze(axis=-1)
        return (
            jnp.sum(
                (jnp.abs(vis - gvis)/data['sigma'])**2 * data['padmask']
            ) / (2*data['padmask'].sum())
        )

    @staticmethod
    def _loss_vis_nfft(vis, data):
        return (
            jnp.sum(
                (jnp.abs(vis - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / (2*data['padmask'].sum())
        )

    @staticmethod
    def _loss_vis_2d(image, data):
        vis = jax.lax.batch_matmul(data['A'][0, ...], image).squeeze(axis=-1)
        return (
            jnp.sum(
                (jnp.abs(vis - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / (2*data['padmask'].sum())
        )

    @staticmethod
    def _loss_amp(video, data, gamp):
        gamp = data['target'] if gamp is None else gamp
        amp = jnp.abs(
            jax.lax.batch_matmul(data['A'][:, 0, ...], video).squeeze(axis=-1)
        )
        return (
            jnp.sum(
                (jnp.abs(amp - gamp)/data['sigma'])**2 * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_amp_nfft(amp, data):
        return (
            jnp.sum(
                (jnp.abs(amp - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_logamp(video, data, gamp):
        gamp = data['target'] if gamp is None else gamp
        data['sigma'] = data['sigma'] / gamp
        gamp = jnp.log(gamp + 1e-12)
        logamp = jnp.abs(
            jax.lax.batch_matmul(data['A'][:, 0, ...], video).squeeze(axis=-1)
        )
        logamp = jnp.log(logamp + 1e-12)
        return (
            jnp.sum(
                (jnp.abs(logamp - gamp)/data['sigma'])**2 * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_logamp_nfft(logamp, data):
        data['sigma'] = data['sigma'] / data['target']
        data['target'] = jnp.log(data['target'] + 1e-12)
        return (
            jnp.sum(
                (jnp.abs(logamp - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_amp_2d(image, data):
        amp = jnp.abs(
            jax.lax.batch_matmul(data['A'][0, ...], image).squeeze(axis=-1)
        )
        return (
            jnp.sum(
                (jnp.abs(amp - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / (2*data['padmask'].sum())
        )

    @staticmethod
    def _loss_cphase(video, data):
        data['target'] = jnp.deg2rad(data['target'])
        data['sigma'] = jnp.deg2rad(data['sigma'])
        vis1 = jax.lax.batch_matmul(
            data['A'][:, 0, ...], video
        ).squeeze(axis=-1)
        vis2 = jax.lax.batch_matmul(
            data['A'][:, 1, ...], video
        ).squeeze(axis=-1)
        vis3 = jax.lax.batch_matmul(
            data['A'][:, 2, ...], video
        ).squeeze(axis=-1)
        cphase = jnp.angle(vis1 * vis2 * vis3)
        return (
            2 * jnp.sum(
                ((1.0 - jnp.cos(cphase - data['target']))/data['sigma']**2)
                * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_cphase_nfft(cphase, data):
        data['target'] = jnp.deg2rad(data['target'])
        data['sigma'] = jnp.deg2rad(data['sigma'])
        return (
            2 * jnp.sum(
                ((1.0 - jnp.cos(cphase - data['target']))/data['sigma']**2)
                * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_cphase_2d(image, data):
        data['target'] = jnp.deg2rad(data['target'])
        data['sigma'] = jnp.deg2rad(data['sigma'])
        vis1 = jax.lax.batch_matmul(data['A'][0, ...], image).squeeze(axis=-1)
        vis2 = jax.lax.batch_matmul(data['A'][1, ...], image).squeeze(axis=-1)
        vis3 = jax.lax.batch_matmul(data['A'][2, ...], image).squeeze(axis=-1)
        cphase = jnp.angle(vis1 * vis2 * vis3)
        return (
            2 * jnp.sum(
                ((1.0 - jnp.cos(cphase - data['target']))/data['sigma']**2)
                * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_bs(video, data):
        vis1 = jax.lax.batch_matmul(
            data['A'][:, 0, ...], video
        ).squeeze(axis=-1)
        vis2 = jax.lax.batch_matmul(
            data['A'][:, 1, ...], video
        ).squeeze(axis=-1)
        vis3 = jax.lax.batch_matmul(
            data['A'][:, 2, ...], video
        ).squeeze(axis=-1)
        bs = vis1 * vis2 * vis3
        return (
            jnp.sum(
                (jnp.abs(bs - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / (2*data['padmask'].sum())
        )

    @staticmethod
    def _loss_bs_2d(image, data):
        vis1 = jax.lax.batch_matmul(data['A'][0, ...], image).squeeze(axis=-1)
        vis2 = jax.lax.batch_matmul(data['A'][1, ...], image).squeeze(axis=-1)
        vis3 = jax.lax.batch_matmul(data['A'][2, ...], image).squeeze(axis=-1)
        bs = vis1 * vis2 * vis3
        return (
            jnp.sum(
                (jnp.abs(bs - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / (2*data['padmask'].sum())
        )

    @staticmethod
    def _loss_logcamp(video, data):
        vis1 = jax.lax.batch_matmul(
            data['A'][:, 0, ...], video
        ).squeeze(axis=-1)
        vis2 = jax.lax.batch_matmul(
            data['A'][:, 1, ...], video
        ).squeeze(axis=-1)
        vis3 = jax.lax.batch_matmul(
            data['A'][:, 2, ...], video
        ).squeeze(axis=-1)
        vis4 = jax.lax.batch_matmul(
            data['A'][:, 3, ...], video
        ).squeeze(axis=-1)
        logcamp = jnp.log(jnp.abs(vis1)) \
                + jnp.log(jnp.abs(vis2)) \
                - jnp.log(jnp.abs(vis3)) \
                - jnp.log(jnp.abs(vis4))
        return (
            jnp.sum(
                (jnp.abs(logcamp - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_logcamp_nfft(logcamp, data):
        return (
            jnp.sum(
                (jnp.abs(logcamp - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_logcamp_2d(image, data):
        vis1 = jax.lax.batch_matmul(data['A'][0, ...], image).squeeze(axis=-1)
        vis2 = jax.lax.batch_matmul(data['A'][1, ...], image).squeeze(axis=-1)
        vis3 = jax.lax.batch_matmul(data['A'][2, ...], image).squeeze(axis=-1)
        vis4 = jax.lax.batch_matmul(data['A'][3, ...], image).squeeze(axis=-1)
        logcamp = jnp.log(jnp.abs(vis1)) \
                + jnp.log(jnp.abs(vis2)) \
                - jnp.log(jnp.abs(vis3)) \
                - jnp.log(jnp.abs(vis4))
        return (
            jnp.sum(
                (jnp.abs(logcamp - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / data['padmask'].sum()
        )

    @staticmethod
    def _loss_mbreve(video, data):
        visI = jax.lax.batch_matmul(
            data['A'][:, 0, ...], video[0]
        ).squeeze(axis=-1)
        visQ = jax.lax.batch_matmul(
            data['A'][:, 0, ...], video[1]
        ).squeeze(axis=-1)
        visU = jax.lax.batch_matmul(
            data['A'][:, 0, ...], video[2]
        ).squeeze(axis=-1)
        mbreve = (visQ + 1j * visU) / visI
        return (
            jnp.sum(
                (jnp.abs(mbreve - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / (2*data['padmask'].sum())
        )

    @staticmethod
    def _loss_mbreve_2d(image, data):
        visI = jax.lax.batch_matmul(
            data['A'][0, ...], image[0]
        ).squeeze(axis=-1)
        visQ = jax.lax.batch_matmul(
            data['A'][0, ...], image[1]
        ).squeeze(axis=-1)
        visU = jax.lax.batch_matmul(
            data['A'][0, ...], image[2]
        ).squeeze(axis=-1)
        mbreve = (visQ + 1j * visU) / visI
        return (
            jnp.sum(
                (jnp.abs(mbreve - data['target'])/data['sigma'])**2
                * data['padmask']
            ) / (2*data['padmask'].sum())
        )

    @staticmethod
    def _loss_lcurve(lcurve, frames):
        return jnp.mean(
            (jnp.sum(jnp.real(frames), axis=1) - lcurve)**2
        )

    @staticmethod
    def _loss_lcurve_2d(lcurve, image):
        return jnp.mean(
            (jnp.sum(jnp.real(image)) - jnp.median(lcurve))**2
        )

    @staticmethod
    def _loss_min_dynamics(dynamic):
        return jnp.mean(
            jnp.sum(jnp.max(dynamic, axis=0))
        )

    @staticmethod
    def _loss_dynamic_flux(dynamic):
        return jnp.mean(
            (jnp.sum(dynamic, axis=1) - jnp.ones(dynamic.shape[0]))**2
        )

    @staticmethod
    def _loss_static_flux(static):
        return jnp.mean((jnp.sum(static) - 1.0)**2)

    @staticmethod
    def _loss_border(frame):
        frame = jnp.real(frame).reshape(-1, NPIX, NPIX)
        pad = NPIX // 20
        return jnp.mean(
            frame[:, 0:pad, :].sum() + frame[:, -pad-1:-1, :].sum()
          + frame[:, :, 0:pad].sum() + frame[:, :, -pad-1:-1].sum()
        )

    @staticmethod
    def _loss_ml_overlap(larr, iarr, tau=0.1):
        return jnp.mean(
            jnp.abs(larr) * jnp.exp(-jnp.abs(iarr) / tau)
        )

    @staticmethod
    def _loss_fn_red(*args, **kwargs):
        loss, (*updates, ldict, video) = Trainer._which_loss_fn(
            *args, **kwargs
        )
        updates = [jax.tree_map(lambda x: jnp.mean(x, axis=0), updates[i])
                    for i in range(len(updates))]
        return loss, (updates, ldict, video)

    @staticmethod
    @jax.jit
    def train_step(kwargs: OrderedDict) -> list[Array]:
        """Training step.

        Args:
            kwargs: Training states and other variables required
                for loss computations. Input as an OrderedDict since
                @jax.jit can change the order in which kwargs are passed.

        Returns:
            Total loss, loss dictionary, sampled video, and training states
        """
        # Unpack states
        keys = list(kwargs.keys())
        states = [kwargs.pop(key) for key in keys if 'state' in key]
        params = [s.params for s in states]
        batch_stats = [s.batch_stats
                       for s in states if s.batch_stats is not None]
        apply_fn = [s.apply_fn for s in states]
        argnums = tuple(range(len(states)))
        args = params + batch_stats + apply_fn
        # Get and apply gradients
        (loss, (updates, ldict, video)), grads = jax.value_and_grad(
            Trainer._loss_fn_red, argnums=argnums, has_aux=True
        )(*args, **kwargs)
        states = [state.apply_gradients(grads=grad)
                  for state, grad in zip(states, grads)]
        # Update batch stats if available
        updatables = [s for s in states if s.batch_stats is not None]
        nonupdatables = [s for s in states if s.batch_stats is None]
        states = [s.replace(batch_stats=u['batch_stats'])
                  for s, u in zip(updatables, updates)] \
               + nonupdatables
        return loss, ldict, *video, *states
