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

"""Data handling and processing. Extends ehtim's Obsdata class."""

import itertools as it
from collections.abc import Callable
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
import scipy.interpolate as interp
import ehtim as eh
import ehtim.imaging.imager_utils as ehu
import ehtim.imaging.pol_imager_utils as ehup

from jax import Array
from jax import numpy as jnp

from . import utils as ut


class Obsdata(eh.obsdata.Obsdata):
    """Extends ehtim's Obsdata class.

    Inherits all attributes and methods from ehtim's Obsdata
    and adds new helpfuls methods for data processing with `kine`.

    See Also:
        eh.obsdata.Obsdata: Base class providing core functionality.
        See https://github.com/achael/eht-imaging/blob/main/ehtim/obsdata.py
    """

    def get_zbl(self) -> float:
        """Get shortest baseline flux density.
        
        Returns:
            Maximum visibility amplitude of shortest baseline.
        """
        # Suppress ehtim's printing
        with ut.no_print():
            self.add_amp(return_type='df')
        # Select shortest baseline
        idx = np.argmin(self.amp['baselength'])
        min_bl = [self.amp.iloc[idx]['t1'], self.amp.iloc[idx]['t2']]
        # Currently using max, prev. was median
        return np.max(self.unpack_bl(min_bl[0], min_bl[1],'amp')['amp'])

    def flag_empty(self) -> Self:
        """Flag sites with no measurements.

        Obsdata sometimes include atennas with no data.
        For instance, after time splitting. Remove those antennas.
        
        Returns:
            Cleared Obsdata object.
        """
        allsites = set(self.unpack(['t1'])['t1']) \
                 | set(self.unpack(['t2'])['t2'])
        self.tarr = self.tarr[[o in allsites for o in self.tarr['site']]]
        # Re-build Obsdata object
        argl, argd = self.obsdata_args()
        return Obsdata(*argl, **argd)

    def norm_to_max(self) -> Self:
        """Normalize amplitudes to shortest baseline flux density.

        Currently required for decomposing
        into static and dynamic components.
        
        Returns:
            Normalized Obsdata object.
        """
        maxamp = self.get_zbl()
        obs = self.switch_polrep('circ')
        for field in [
            'rrvis',
            'llvis',
            'rlvis',
            'lrvis',
            'rrsigma',
            'llsigma',
            'rlsigma',
            'lrsigma'
        ]:
            obs.data[field] /= maxamp
        obs = obs.switch_polrep('stokes')
        # Re-build Obsdata object
        argl, argd = obs.obsdata_args()
        return Obsdata(*argl, **argd)

    def fix_multiepoch(self, refobs: Self) -> None:
        """Fix metadata for multi-epoch dynamic imaging.
        
        Given a reference Obsdata object, change metadata of
        current Obsdata so they match. Useful for multi-epoch imaging.

        Args:
            refobs: Reference Obsdata object.
        """
        self.source = refobs.source
        self.ra = refobs.ra
        self.dec = refobs.dec
        self.rf = refobs.rf
        self.bw = refobs.bw
        self.timetype = refobs.timetype
        self.polrep = refobs.polrep
    
    def avg_coherent(self, tavg: float, scan_avg: bool = False) -> Self:
        """Wrapper for ehtim's coherent time-averaging.
        
        Args:
            tavg: Averaging time.
            scan_avg: Whether to scan average data.
        
        Returns:
            Coherently time-averaged Obsdata object.
        """
        obs = super().avg_coherent(tavg, scan_avg=scan_avg)
        # Re-build Obsdata object
        argl, argd = obs.obsdata_args()
        return Obsdata(*argl, **argd)
    
    def add_fractional_noise(self, frac: float, debias: bool = False) -> Self:
        """Wrapper for ehtim's fractional noise addition.
                
        Args:
            frac: Noise percentage to be added.
            debias: Whether or not to add frac of debiased amplitudes.
        
        Returns:
            Noise-inflated Obsdata object.
        """
        obs = super().add_fractional_noise(frac, debias=debias)
        # Re-build Obsdata object
        argl, argd = obs.obsdata_args()
        return Obsdata(*argl, **argd)

    def flag_UT_range(
            self,
            UT_start_hour: float = 0.0,
            UT_stop_hour: float = 0.0,
            flag_type: str = 'all',
            flag_what: str = '',
            output: str = 'kept'
    ) -> Self:
        """Wrapper for ehtim's flat UT range.
                        
        Args:
            UT_start_hour: Start of time window.
            UT_stop_hour: End of time window.
            flag_type: 'all', 'baseline', or 'station'.
            flag_what: Baseline or station to flag.
            output: Returns 'kept', 'flagged', or 'both' (a dictionary).
        
        Returns:
            Time-flagged Obsdata object.
        """
        obs = super().flag_UT_range(
            UT_start_hour=UT_start_hour,
            UT_stop_hour=UT_stop_hour,
            flag_type=flag_type,
            flag_what=flag_what,
            output=output
        )
        # Re-build Obsdata object
        argl, argd = obs.obsdata_args()
        return Obsdata(*argl, **argd)
    
    def flag_uvdist(
            self,
            uv_min: float = 0.0,
            uv_max: float = 1e12,
            output: str = 'kept'
    ) -> Self:
        """Wrapper for ehtim's uv distance flagging.
                                
        Args:
            uv_min: Remove points with uvdist less than this.
            uv_max: Remove points with uvdist greater than this.
            output: Returns 'kept', 'flagged', or 'both' (a dictionary).
        
        Returns:
            uv-distance-flagged Obsdata object.
        """
        obs = super().flag_uvdist(
            uv_min=uv_min,
            uv_max=uv_max,
            output=output
        )
        # Re-build Obsdata object
        argl, argd = obs.obsdata_args()
        return Obsdata(*argl, **argd)

    def flag_sites(self, sites: list, output: str = 'kept') -> Self:
        """Wrapper for ehtim's site flagging.
                                        
        Args:
            sites: List of sites to remove from the data.
            output: Returns 'kept', 'flagged', or 'both' (a dictionary).
        
        Returns:
            Site-flagged Obsdata object.
        """
        obs = super().flag_sites(
            sites=sites,
            output=output
        )
        # Re-build Obsdata object
        argl, argd = obs.obsdata_args()
        return Obsdata(*argl, **argd)
    
    def flag_bl(self, sites: list, output: str = 'kept') -> Self:
        """Wrapper for ehtim's baseline flagging.
                                                
        Args:
            sites: Baseline to remove from the data.
            output: Returns 'kept', 'flagged', or 'both' (a dictionary).
        
        Returns:
            Baseline-flagged Obsdata object.
        """
        obs = super().flag_bl(
            sites=sites,
            output=output
            )
        # Re-build Obsdata object
        argl, argd = obs.obsdata_args()
        return Obsdata(*argl, **argd)
    
    def split_obs(
            self,
            t_gather: float = 0.0,
            scan_gather: bool = False,
            min_bl: int = 0,
            group: int = 0
    ) -> list:
        """Split observation wrapper to allow for 'grouping'
        
        Args:
            t_gather: Snapshot duration (in seconds).
            scan_gather: If true, gather data into scans.
            min_bl: Minimum number of baselines allowed per snapshot.
            group: Number of adjacent snapshot to group.
        
        Returns:
            List of snapshot Obsdata objects.
        """
        # Suppress ehtim's printing
        with ut.no_print():
            glist = super().split_obs(
                t_gather=t_gather,
                scan_gather=scan_gather
            )
            glist = [Obsdata(*ob.obsdata_args()[0], **ob.obsdata_args()[1])
                     for ob in glist]
        # Drop snapshots with less baselines than specified
        if min_bl > 0:
            nvis = min_bl * (min_bl-1) / 2
            glist = [ob for ob in glist if len(ob.data) >= nvis]
        # Group observations
        if group > 0:
            # Pad list of observations
            ini = [glist[0]] * group
            fin = [glist[-1]] * group
            glist = ini + glist + fin
            # Group multiple observations
            glist = [glist[i-group : i+group+1]
                     for i in np.arange(group, len(glist)-group)]
            glist = [self.merge_obs(g) for g in glist]
        print(f'Splitting Observation File into {len(glist)} times')
        return glist
    
    def get_lightcurve(
            self,
            tavg: float = 0.0,
            min_bl: int = 0,
            uv_max: float | None = None,
            times: list | None = None
    ) -> Array:
        """Extract interpolated light curve from intra-site baselines.

        Args:
            tavg: Snapshot duration (in seconds).
            min_bl: Minimum number of baselines per snapshot.
            uv_max: Maximum uv-distance allowed for flux density extraction.
            times: Time stamps over which interpolate (optional).

        Returns:
            Interpolated light-curve array.
        """
        with ut.no_print():
            self.add_scans()
            obs_split = self.split_obs(t_gather=tavg, min_bl=min_bl)
        # Univariate spline interpolation
        alltimes = np.array(
            [obs_split[j].data['time'][0] for j in range(len(obs_split))]
        )
        if uv_max is None:
            allfluxes = np.array(
                [np.median(obs_split[j].get_zbl())
                 for j in range(len(obs_split))]
            )
        else:
            allfluxes = np.array(
                [np.median(obs_split[j].flag_uvdist(uv_max=uv_max)\
                                       .unpack('amp')['amp'])
                for j in range(len(obs_split))]
            )
        # Sort by time
        idxsort = np.argsort(alltimes)
        alltimes = alltimes[idxsort]
        allfluxes = allfluxes[idxsort]
        # NaN masking
        mask = np.isnan(allfluxes)
        maskedtimes = alltimes[~mask]
        maskedfluxes = allfluxes[~mask]
        # Interpolation
        spl = interp.UnivariateSpline(maskedtimes, maskedfluxes, ext=3)
        spl.set_smoothing_factor(1e-10)
        spl_times = alltimes
        if times is not None:
            spl_times = times
        spl_fluxes = spl(spl_times)
        return ut.list_to_jaxarr(spl_fluxes)
    
    @staticmethod
    def get_data(
            obs: Self | list[Self],
            dtype: str,
            prior: eh.image.Image,
            ttype: str = 'direct'
    ) -> list[Array]:
        """Generate data products.

        Args:
            obs: Obsdata or list of Obsdata from which
                data products are computed.
            dtype: Data product type (e.g., visI, cphaseI).
                Last letter indicates polarization.
            prior: ehtim's Image object from which metadata is extracted.
            ttype: Fourier transform type ('direct', 'nfft').

        Returns:
            List of jnp.ndarrays with data products and corresponding sigmas,
            Fourier trnasformations (optional) and padding masks (optional).
        """
        # Check whether single or list of obs
        if not isinstance(obs, list):
            # mbreve data products
            if dtype == 'mbreve':
                target, sigma, A = ehup.chisqdata_m(obs, prior, mask=[])
            # Other data products
            else:
                target, sigma, A = ehu.chisqdata(
                    obs,
                    prior,
                    mask=[],
                    dtype=dtype[:-1],
                    pol=dtype[-1],
                    ttype=ttype,
                    debias=False,
                    maxset=False
                )
            # Closures data products
            target, sigma = target.reshape(1, -1), sigma.reshape(1, -1)
            if not isinstance(A, tuple):
                A = A.reshape(1, A.shape[0], A.shape[1])
            return ut.list_to_jaxarr(target, sigma, A)

        # Extract data
        target, sigma, A, padmask = [], [], [], []
        for i, _ in enumerate(obs):
            # mbreve data products
            if dtype == 'mbreve':
                t, s, a = ehup.chisqdata_m(obs[i], prior, mask=[])
            # Other data products
            else:
                t, s, a = ehu.chisqdata(
                    obs[i],
                    prior,
                    mask=[],
                    dtype=dtype[:-1],
                    pol=dtype[-1],
                    ttype=ttype,
                    debias=False,
                    maxset=False
                )
            # Closures data products
            if not isinstance(a, tuple):
                a = a.reshape(1, a.shape[0], a.shape[1])
            target.append(t)
            sigma.append(s)
            A.append(jnp.array(a))
            padmask.append(jnp.ones(len(t)))
        # Helper padding function
        def pad(data, maxv):
            if len(data.shape) < 3:
                return jnp.ones((maxv-data.shape[0],)+data.shape[1:])
            return jnp.ones(
                    (data.shape[0], maxv-data.shape[1],)
                    + data.shape[2:]
                   )
        # Pad data
        maxv = np.max([len(t) for t in target])
        for i in range(len(obs)):
            target[i] = jnp.concatenate(
                [target[i], pad(target[i], maxv)],
                axis=0
            )
            sigma[i] = jnp.concatenate(
                [sigma[i], pad(sigma[i], maxv)],
                axis=0
            )
            padmask[i] = jnp.concatenate(
                [padmask[i], pad(padmask[i], maxv) * 0],
                axis=0
            )
            A[i] = jnp.concatenate([A[i], pad(A[i], maxv)], axis=1)
        return ut.list_to_jaxarr(target, sigma, A, padmask)

    @staticmethod
    def get_data_nfft(
            obs: Self | list[Self],
            dtype: str,
            prior: eh.image.Image,
            ttype: str = 'nfft'
    ) -> list[Array]:
        """Generate data products with NUFFT.

        Args:
            obs: Obsdata or list of Obsdata from which
                data products are computed.
            dtype: Data product type (e.g., visI, cphaseI).
                Last letter indicates polarization.
            prior: ehtim's Image object from which metadata is extracted.
            ttype: Fourier transform type ('direct', 'nfft').

        Returns:
            List of jnp.ndarrays with data products and corresponding sigmas,
            Fourier trnasformations (optional) and padding masks (optional).
        """
        # Check whether single or list of obs
        if not isinstance(obs, list):
            target, sigma, _ = ehu.chisqdata(
                obs,
                prior,
                mask=[],
                dtype=dtype[:-1],
                pol=dtype[-1],
                ttype=ttype,
                debias=False,
                maxset=False
            )
            del _
            # closures data products
            target, sigma = target.reshape(1, -1), sigma.reshape(1, -1)
            return ut.list_to_jaxarr(target, sigma)

        # Extract data
        target, sigma, padmask = [], [], []
        for i, _ in enumerate(obs):
            t, s, _ = ehu.chisqdata(
                obs[i],
                prior,
                mask=[],
                dtype=dtype[:-1],
                pol=dtype[-1],
                ttype=ttype,
                debias=False,
                maxset=False
            )
            del _
            # closures data products
            target.append(t)
            sigma.append(s)
            padmask.append(jnp.ones(len(t)))
        # Helper padding function
        def pad(data, maxv):
            return jnp.ones((maxv-data.shape[0],)+data.shape[1:])
        # Pad data
        maxv = np.max([len(t) for t in target])
        for i in range(len(obs)):
            target[i] = jnp.concatenate(
                [target[i], pad(target[i], maxv)],
                axis=0
            )
            sigma[i] = jnp.concatenate(
                [sigma[i], pad(sigma[i], maxv)],
                axis=0
            )
            padmask[i] = jnp.concatenate(
                [padmask[i], pad(padmask[i], maxv) * 0],
                axis=0
            )
        return ut.list_to_jaxarr(target, sigma, padmask)
    
    @staticmethod
    def _get_baselines(obslist: list, conj: bool = False) -> NDArray:
        """Retrieve baselines codenames.

        Args:
            obslist: List of snapshot Obsdata objects.
            conj: If true, return conjugate baselines as well.

        Returns:
            Array of baselines codenames.
        """
        # Helper padding function
        def pad(data, maxv):
            return [
                tuple(x)
                for x in np.full((maxv-len(data), 2), fill_value='pad')
            ]
        # Extract baseline names
        blines = []
        for obs in obslist:
            blines.append(list(obs.unpack(['t1', 't2'], conj=conj)))
        # Pad data
        maxv = np.max([len(b) for b in blines])
        for i, _ in enumerate(blines):
            blines[i] = blines[i] + pad(blines[i], maxv)
        return np.array(blines)

    @staticmethod
    def _site_to_index(sites: dict, blines: np.ndarray) -> Array:
        """Convert baselines codenames to baseline indices.

        Args:
            sites: Dictionary of sites and corresponding index.
            blines: Array of baselines codenames.

        Returns:
            Array of baselines indices.
        """
        indices = [
            [tuple(sites[b] for b in bs) for bs in bls]
            for bls in blines
        ]
        return jnp.array(indices)

    def set_gains_vars(self, obslist: list, gains_prior: dict) -> list:
        """Set variables needed for gain fitting.

        Args:
            obslist: List of snapshot Obsdata objects.
            gains_prior: Per-site allowed ranges for amplitude gains values.

        Returns:
            List of variable required for simultaneous gain fitting.
        """
        # Get antenna codenames and index
        sites = self.tkey
        # Get number of visibilities
        nvis = len(sites) * (len(sites) - 1) // 2
        # Update sites dict with a padding antenna for vectorization
        sites.update({'pad': len(sites)})
        # Get number of antennas (including pad)
        nsites = len(sites)
        # Get baselines codenames and indices
        blname = self._get_baselines(obslist)
        blindx = self._site_to_index(sites, blname)
        # Get gains prior per antenna
        gains_prior.update({'pad': [1.00, 1.00]})
        gains_range = np.array([gains_prior[site] for site in sites])
        lower = jnp.array(gains_range[:, 0])
        upper = jnp.array(gains_range[:, 1])
        return sites, nsites, nvis, blindx, lower, upper
    
    def get_baselines_nfft(self, conj: bool = True) -> list:
        """Retrieve baselines codenames.

        Args:
            conj: If true, return conjugate baselines as well.

        Returns:
            List of baselines codenames.
        """
        # Extract baseline names
        tlist = self.tlist(conj=conj)
        blines = []
        for tdata in tlist:
            blines.append([(dat['t1'], dat['t2']) for dat in tdata])
        return blines
    
    def get_uvpoints(self, psize: float, conj: bool = True) -> dict:
        """Get and scale uv coordinates for NUFFT computations.

        Args:
            psize: Pixel size in radians.
            conj: If true, return conjugate baselines as well.

        Returns:
            Scaled uv coordinates.
        """
        # Extract uv coordinates
        tlist = self.tlist(conj=conj)
        us, vs = [], []
        for tdata in tlist:
            us.append([dat['u'] * psize * 2 * np.pi for dat in tdata])
            vs.append([dat['v'] * psize * 2 * np.pi for dat in tdata])
        return {'u': np.concatenate(us), 'v': np.concatenate(vs)}

    def get_pulsefac(self, uv: dict, pulse: Callable) -> NDArray:
        """Get pulse factors for NUFFT computations.

        Args:
            uv: uv coordinates.
            pulse: pulse function from ehtim's collection.

        Returns:
            Pulses.
        """
        pulsefac = []
        for u, v in zip(uv['u'], uv['v']):
            phases = np.exp(-1j * 0.5 * (u + v))
            pulses = pulse(u, v, 1., dom='F')
            pulsefac.append(pulses * phases)
        return np.array(pulsefac)

    @staticmethod
    def _tri_minimal_set(sites, tarr):
        """
        Returns a minimal set of triangles for bispectra and closure phases
        (adapted from ehtim.observing.obs_helpers.tri_minimal_set)
        """
        # Determine ordering and reference site based on order of self.tarr
        sites_ordered = [x for x in tarr['site'] if x in sites]
        ref = sites_ordered[0]
        sites_ordered.remove(ref)
        # Find all triangles that contain the ref
        tris = list(it.combinations(sites_ordered, 2))
        return [[(ref, t[0]), (t[0], t[1]), (t[1], ref)] for t in tris]

    @staticmethod
    def _quad_minimal_set(sites, tarr):
        """
        Returns a minimal set of quadrangels for closure amplitudes
        (adapted from ehtim.observing.obs_helpers.quad_minimal_set)
        """
        # Determine ordering and reference site based on order of  self.tarr
        sites_ordered = np.array([x for x in tarr['site'] if x in sites])
        ref = sites_ordered[0]
        # Loop over other sites >=3 and form minimal closure amplitude set
        quads = []
        for i in range(3, len(sites_ordered)):
            for j in range(1, i):
                if j == i-1:
                    k = 1
                else:
                    k = j+1
                # Convention is (12)(34)/(14)(23)
                quad = [
                    (ref, sites_ordered[i]),
                    (sites_ordered[j], sites_ordered[k]),
                    (ref, sites_ordered[k]),
                    (sites_ordered[i], sites_ordered[j])
                ]
                quads.append(quad)
        return quads

    def get_closure_baselines(self, which: str) -> list:
        """Get minimal set of closure phases or closure amplitudes.

        Args:
            which: 'triangles' for closure phases,
                'quadrangles' for closure amplitudes.

        Returns:
            Closure phases or closure amplitudes codenames.
        """
        tlist = self.tlist(conj=True)
        closures = []
        for tdata in tlist:
            sites = list(set(np.hstack((tdata['t1'], tdata['t2']))))
            if which == 'triangles':
                closures.append(self._tri_minimal_set(sites, self.tarr))
            elif which == 'quadrangles':
                closures.append(self._quad_minimal_set(sites, self.tarr))
        return closures
    
    def get_closure_indices(
            self,
            blname: list,
            which: str = 'triangles'
    ) -> NDArray:
        """Get indices of baselines forming closure products.
        
        Match baselines in triangles or quadrangles with ordered baselines
        and return indices for closure data product construction.

        Args:
            blname: Baselines codenames.
            which: 'triangles' for closure phases,
                'quadrangles' for closure amplitudes.
        
        Returns:
            Closure product baselines indices.
        """
        # Retrieve closure baselines
        clarr = self.get_closure_baselines(which=which)
        # Iterate over scans
        frame_indices, pad = [], 0
        for i, _ in enumerate(blname):
            # Skip empty closures (couldn't form a cphase/camp)
            if len(clarr[i]) == 0:
                pad += len(blname[i])
                continue
            # Get baseline indices for the current fram
            blindex = {tuple(bl): idx for idx, bl in enumerate(blname[i])}
            # Loop over each triangle/quadrangle
            scan_indices = []
            for closure in clarr[i]:
                # Loop over each combination in the trio
                closure_indices = []
                for bl in closure:
                    # Get the index from the dictionary and append it
                    # to the closure's indices
                    if tuple(bl) in blindex:
                        index = blindex[tuple(bl)] + pad
                    else:
                        index = -1
                    closure_indices.append(index)
                # Append the closure indices to the frame indices
                if -1 in closure_indices:
                    continue
                scan_indices.append(closure_indices)
            pad += len(blname[i])
            # Check scan_indices is not empty
            if scan_indices:
                frame_indices.append(scan_indices)
        return np.concatenate(frame_indices)

    @classmethod
    def load_uvfits(cls, inpath: str, **kwargs) -> Self:
        """Wrapper for ehtim's obsdata loading function.

        Args:
            inpath: uvfits path.
            **kwargs: ehtim's load_uvfits keyword arguments.

        Returns:
            Obsdata object.
        """
        obs = eh.obsdata.load_uvfits(inpath, **kwargs)
        argl, argd = obs.obsdata_args()
        return cls(*argl, **argd)

    @classmethod
    def merge_obs(cls, obslist: list[Self], **kwargs) -> Self:
        """Wrapper for ehtim's merge obs function.

        Args:
            obslist: Snapshot Obsdata to be merged.
            **kwargs: ehtim's merge_obs keyword arguments.
        
        Returns:
            Merged Obsdata object.
        """
        obs = eh.obsdata.merge_obs(obslist, **kwargs)
        argl, argd = obs.obsdata_args()
        return cls(*argl, **argd)
    