"""
Provide interfaces to handle hyper-images.

Consider the NPHyperImage class to efficiently perform operations on the entire
hyper-image. Each file class can be casted to NPHyperImage.

When using a large NC file, consider importing it with NCHyperImage and removing
useless data before importing it as NPHyperImage.
"""

from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path
import spectral
import xarray

import utils.spectroscopy.signal_operations as signop


####====------------------------  Parameters  --------------------------====####

class SpectralType:
    LIBS = 0
    SWIR = 1

class ComparisonMethod:
    """
    - SSE = Sum of squared errors
    - SAM = Spectral angle mapping
    """
    SSE = 0
    SAM = 1


####====--------------------  General hyper-image  ---------------------====####

class HyperImage(ABC):

    @property
    @abstractmethod
    def sp_type(self) -> int:
        """Data type of the hyperspectral image, see SpectralType class for
        available types."""
        ...

    @property
    @abstractmethod
    def axis_id_x(self): ...

    @property
    @abstractmethod
    def axis_id_y(self): ...

    @property
    @abstractmethod
    def axis_id_bands(self): ...

    @property
    @abstractmethod
    def nbr_bands(self):
        """Return the number of bands."""
        ...

    @property
    @abstractmethod
    def shape(self):
        """Spatial shape as (sx, sy)."""
        ...

    @property
    @abstractmethod
    def bands(self):
        """
        Returns
        -------
        array-like
            Array of band centers
        """
        ...

    @abstractmethod
    def trim_bands_keep(self, *kept_intervals):
        """Reduce the band domain to the intervals. Interval bounds are absolute
        values of spectral axis."""
        ...

    @abstractmethod
    def trim_bands_remove(self, *removed_intervals):
        """Remove intervals from the band domain. Interval bounds are absolute
        values of spectral axis."""
        ...

    @abstractmethod
    def trim_spatial_keep(self, *intervals, axis):
        """Reduce the spatial domain to the intervals. Interval bounds are
        relative (in [0, 1]) w.r.t. the spatial size of the corresponding axis.
        """
        ...

    @abstractmethod
    def trim_spatial_remove(self, *intervals, axis):
        """Remove the intervals from the spatial domain. Interval bounds are
        relative (in [0, 1]) w.r.t. the spatial size of the corresponding axis.
        """
        ...

    @abstractmethod
    def downsample(self, n, axis):
        """Downsampling of the hyperimage such that only n-th sample are kept
         along the axis."""
        ...

    @abstractmethod
    def pixel_spectrum(self, coords: tuple[int, int]) -> np.ndarray:
        """Get the spectrum intensities at given pixel.

        Parameters
        ----------
        coords: tuple[int, int]
            The pixel coordinates (x, y)
        Returns
        -------
        array-like
            Signal intensities
        """
        ...

    def rloc_spectrum(self, rloc: tuple[float, float]) -> np.ndarray:
        """Get the spectrum intensities at given relative location.

        Parameters
        ----------
        rloc: tuple[float, float]
            Relative location of the pixel to extract the spectrum from
        Returns
        -------
        array-like
            Signal intensities
        """
        assert 0 <= rloc[0] <= 1
        assert 0 <= rloc[1] <= 1
        return self.pixel_spectrum(self._loc_rel2abs(rloc))

    @abstractmethod
    def band_map(self, band):
        """Get the pixel intensities at given band (closest will be selected).

        Returns
        -------
        np.ndarray:
            2-D image
        """
        ...

    @abstractmethod
    def similarity_map(self, *signals_ref: np.array,
                       method: int = ComparisonMethod.SAM):
        """Compare the signal of each pixel of the hyperimage with several
        references.

        Band sampling is assumed the same between the spectra. Reference signals
        are standardized (SMV).

        Parameters
        ----------
        signals_ref: list[np.ndarray]
            Reference signal spectra to be compared with those of each pixel
        method: int
            Method used to perform the comparison between spectra,
            see ComparisonMethod class for available options.

        Returns
        -------
        np.ndarray
            2-D similarity map (arbitrary units) with one channel per signal
            reference (shape = (Nx, Ny, Nref))
        """
        ...

    def find_closest_band(self, band):
        """Get index of the `self.bands` value closest to `band`."""
        bands = self.bands
        if band <= bands[0]: return 0
        if band >= bands[-1]: return len(bands)-1
        for i in range(1, len(bands)):
            if band < bands[i]:
                if (bands[i] - band) < (band - bands[i-1]):
                    return i
                else:
                    return i-1
        raise Exception('Implementation error: closest not found')

    def _loc_rel2abs(self, loc: tuple[float, float]):
        """Convert relative location to pixel coordinates."""
        sx, sy = self.shape
        lx, ly = loc
        return int(sx*lx), int(sy*ly)

    def _loc_abs2rel(self, coords: tuple[int, int]):
        """Convert pixel coordinates to relative location."""
        sx, sy = self.shape
        x, y = coords
        return x/sx, y/sy

    def _spatial_rel2abs(self, *rvals: float, axis: int):
        """Convert relative values to absolute ones along axis 0 (x) or 1 (y)."""
        assert axis in (0, 1)
        vmax = self.shape[axis] - 1
        return [int(rval * vmax) for rval in rvals]

    def _spatial_abs2rel(self, *vals: int, axis: int):
        """Convert absolute values to relative ones along axis 0 (x) or 1 (y)."""
        assert axis in (0, 1)
        vmax = self.shape[axis] - 1
        return [val/vmax for val in vals]

class NPHyperImage(HyperImage):
    """HyperImage which is loaded to memory as numpy array to optimize methods
    on the entire hyperimage.

    Optimization for storage and computation:
    - np.array should be stored in memory as np.float32
    - np.array should be saved to disk as np.float16
    """

    @property
    def memory_size(self):
        """The memory size of the hyper-image in Gb."""
        return self._him.size * self._him.itemsize * 1e-9

    @property
    def dtype(self): return self._him.dtype

    @property
    def sp_type(self): return self._sp_type

    @property
    def axis_id_x(self): return 0

    @property
    def axis_id_y(self): return 1

    @property
    def axis_id_bands(self): return 2

    @property
    def nbr_bands(self): return len(self._bands)

    @property
    def shape(self): return self._him.shape[0:2]

    @property
    def bands(self) -> np.ndarray: return self._bands

    @property
    def data_size(self):
        """Return size of the hyper-image data as Gb."""
        return self._him.itemsize * self._him.size * 1e-9

    def __init__(self, bands: np.ndarray, him: np.ndarray,
                 sp_type: int, dtype: np.dtype = np.float32):
        """
        Parameters
        ----------
        bands: np.ndarray
            Band values of the spectra
        him: np.ndarray
            Hyper-image shaped as (nbr_x, nbr_y, nbr_bands)
        sp_type: int
            Spectral type, see SpectralType class for available types.
        dtype: np.dtype
            Data type of the hyper-image element stored in the class
        """
        assert len(him.shape) == 3,\
            f'Should be 3 dimensional, not {len(him.shape)}.'
        self._bands = bands
        if dtype != him.dtype:
            print(f'Convert data type from {him.dtype} to {dtype}...', end='')
            self._him = him.astype(dtype)
            print('   done!')
        else:
            self._him = him
        self._sp_type = sp_type

    ##==--------------------  Decrease dimensionality  ---------------------==##

    def downsample(self, n, axis):
        assert axis in (self.axis_id_x, self.axis_id_y, self.axis_id_bands)

        if axis == self.axis_id_bands:
            self._bands = self._bands[::n]
            self._him = self._him[:, :, ::n]
        elif axis == self.axis_id_x:
            self._him = self._him[::n, :, :]
        else:  # axis=self.axis_id_y
            self._him = self._him[:, ::n, :]

    def trim_bands_keep(self, *kept_intervals):
        ids_keep = signop.ids_inside(self._bands, *kept_intervals)
        self._bands = self._bands[ids_keep]
        self._him = self._him[:, :, ids_keep]

    def trim_bands_remove(self, *removed_intervals):
        ids_keep = signop.ids_outside(self._bands, *removed_intervals)
        self._bands = self._bands[ids_keep]
        self._him = self._him[:, :, ids_keep]

    def trim_spatial_keep(self, *intervals, axis):
        assert axis in (self.axis_id_x, self.axis_id_y)

        # Convert relative intervals to absolute pixel locations
        intervals = [self._spatial_rel2abs(*v, axis=axis) for v in intervals]

        ids_keep = list(set().union(*[range(v[0], v[-1]) for v in intervals]))
        ids_keep = [i for i in ids_keep if i < self.shape[axis]]

        if axis == self.axis_id_x:
            self._him = self._him[ids_keep, :, :]
        else:  # axis=self.axis_id_y
            self._him = self._him[:, ids_keep, :]

    def trim_spatial_remove(self, *intervals, axis):
        assert axis in (self.axis_id_x, self.axis_id_y)

        # Convert relative intervals to absolute pixel locations
        intervals = [self._spatial_rel2abs(*v, axis=axis) for v in intervals]

        ids = np.arange(self.shape[axis], dtype=int)
        ids_remove = list(set().union(*[range(v[0], v[-1]) for v in intervals]))
        ids_keep = [i for i in ids
                    if i not in ids_remove and i < self.shape[axis]]

        if axis == self.axis_id_x:
            self._him = self._him[ids_keep, :, :]
        else:  # axis=self.axis_id_y
            self._him = self._him[:, ids_keep, :]

    ##==------------------------  Decrease storage  ------------------------==##

    def convert_dtype(self, dtype: np.dtype):
        """Change the dtype of the numpy data."""
        print(f'Convert data type from {self._him.dtype} to {dtype}...', end='')
        self._him = self._him.astype(dtype)
        print('   done!')

    ##==------------------------  Spectral scaling  ------------------------==##

    def normalize(self):
        """Scale each pixel spectrum btw 0 and 1."""
        normalize_pixels(self._him, 1)

    def standardize(self):
        """Standardize each pixel spectrum to have zero-mean and unitary
        standard deviation."""
        standardize(self._him)

    ##==--------------------  Spectral pre-processing  ---------------------==##

    def remove_baselines(self):
        him: np.ndarray = self._him

        # Find baselines
        it = np.nditer(him, op_axes=((0, 1),), flags=['multi_index'])
        for i, _ in enumerate(it):
            if i % 10000 == 0: print(f'{i}/{self.shape[0]*self.shape[1]}')
            x, y = it.multi_index
            signal = him[x, y, :]
            baseline = signop.find_convex_hull(signal)
            # TODO: adapt for LIBS spectra with remove_baseline method
            max_baseline = np.max(baseline)
            him[x, y, :] = max_baseline + signal - baseline

    ##==--------------------  Access spectra and maps  ---------------------==##

    def pixel_spectrum(self, coords: tuple[int, int]) -> np.ndarray:
        assert 0 <= coords[0] <= self.shape[0]
        assert 0 <= coords[1] <= self.shape[1]
        x, y = coords
        return self._him[x, y]

    def band_map(self, band):
        band_index = self.find_closest_band(band)
        return self._him[:, :, band_index]

    def similarity_map(self, *signals_ref: np.array,
                       method: int = ComparisonMethod.SAM):

        # Spectral angle mapping
        if method == ComparisonMethod.SAM:
            # SAM method does not require scaling of spectra1

            return [similarity_map_sam(self._him, s) for s in signals_ref]

        # Sum of squared errors
        elif method == ComparisonMethod.SSE:
            # Make spectral compatible for comparison
            signals_ref = [signop.standardization(y) for y in signals_ref]
            him = standardized(self._him)

            return [similarity_map_sse(him, s, skip_scaling=True)
                    for s in signals_ref]

        # Unkown method
        else:
            raise Exception('Unkown similarity map method!')

    ##==-------------------  Cast from file extension  ---------------------==##

    @staticmethod
    def from_any_file(path_file: str, stype: int):
        """Cast file to NPHyperImage from its extension."""
        suffix = Path(path_file).suffix
        if suffix == '.nc':
            return NPHyperImage._from_nc_file(path_file, stype)
        elif suffix == '.hdr':
            return NPHyperImage._from_hdr_file(path_file, stype)
        else:
            raise Exception('Unkown hyper-image file extension')

    @staticmethod
    def _from_hdr_file(path_hdrfile: str, stype: int = SpectralType.SWIR):
        # Import as np.array
        dataset = spectral.open_image(path_hdrfile)
        him: np.ndarray = dataset[:, :, :]  # (nrows, ncols, nbands)
        bands = np.array(dataset.bands.centers)
        return NPHyperImage(bands, him, stype)

    @staticmethod
    def _from_nc_file(path_ncfile: str, stype: int = SpectralType.LIBS):
        ds = xarray.open_dataset(str(path_ncfile))
        ds.load()

        # Ensure that x-axis is the short one when converting to numpy array
        sx, sy = ds.sizes['x'], ds.sizes['y']
        if sx < sy:
            ds = ds.transpose('x', 'y', 'bands')
        else:
            ds = ds.transpose('y', 'x', 'bands')

        bands = np.array(ds['bands'])
        him = ds.to_array().to_numpy()[0]
        return NPHyperImage(bands, him, stype)


class CompressedHyperImage(NPHyperImage):
    """Compressed NPHyperImage where each pixel spectrum sampling has been
     compressed using Visvalingam–Whyatt algorithm."""

    # TODO: this class does not need to remove values of self._him and should
    #       only retain band_vw_masks

    def __init__(self, bands: np.ndarray, hyperimage: np.ndarray,
                 sp_type: int, sampling_ratio: float):
        """
        Parameters
        ----------
        bands: np.ndarray
            Band values of the spectra
        hyperimage: np.ndarray
            Hyper-image shaped as (nbr_x, nbr_y, nbr_bands)
        sp_type: int
            Spectra type, see SpectralType class for available types.
        sampling_ratio: float
            The ratio of spectral point to keep
        """
        super().__init__(bands, hyperimage, sp_type)
        self._bands = bands
        self._sp_type = sp_type
        self.sampling_ratio = sampling_ratio
        self._him, self._band_masks = self.__compress(
            hyperimage, bands, sampling_ratio)
        print(self._him.shape)

    @staticmethod
    def __compress(him, bands, ratio: float):
        """Compress the hyperimage data using Visvalingam–Whyatt algorithm.

        Parameters
        ----------
        him: np.ndarray
            Hyperimage array of the intensities with shape (Nx, Ny, Nbands)
        bands: np.array
            Bands
        ratio: float
            The ratio of spectral point to keep
        """
        band_masks = np.ones(him.shape, dtype=bool)
        sx, sy = him.shape[0:2]
        for i in range(sx):
            if i % 4 == 0:
                print(f'{i*sy}/{sx * sy}')
            for j in range(sy):
                mask = signop.reduce_sampling_vw(bands, him[i, j], ratio)
                band_masks[i, j, :] = mask

        him = him[band_masks].reshape((sx, sy, -1))
        return him, band_masks

    def pixel_bands(self, coords: tuple[int, int]) -> np.ndarray:
        """Get the spectrum bands at given pixel.

        Parameters
        ----------
        coords: tuple[int, int]
            The pixel coordinates (x, y)
        Returns
        -------
        array-like
            Spectrum bands
        """
        x, y = coords
        return self.bands[self._band_masks[x, y]]

    def band_map(self, band):
        """Deprecated"""
        return np.nanstd(self._him, axis=self.axis_id_bands)


####====-----------------  File-specific hyper-image  ------------------====####

class NCHyperImage(HyperImage):
    @property
    def sp_type(self): return self._sp_type

    @property
    def axis_id_x(self): return 0

    @property
    def axis_id_y(self): return 1

    @property
    def axis_id_bands(self): return 2

    @property
    def nbr_bands(self) -> int: return self.dataset.sizes['bands']

    @property
    def shape(self): return self.dataset.sizes['x'], self.dataset.sizes['y']

    @property
    def bands(self): return self._bands

    def __init__(self, path_ncfile: str, sp_type: int = SpectralType.LIBS,
                 load_into_RAM=False):
        """
        Parameters
        ----------
        path_ncfile: str
            Location of the .nc file dataset
        load_into_RAM: bool
            Load the dataset to RAM to increase access time
        """

        self.load_into_RAM = load_into_RAM
        self._sp_type = sp_type

        # Import dataset
        self.dataset = xarray.open_dataset(str(path_ncfile))
        self._bands = np.array(self.dataset['bands'])

        # To prevent from duplicated band, what might cause issue in
        # self.dataset.drop_isel() method
        perturb = np.random.random(len(self._bands)) * 1e-9
        self.dataset['bands'] = self._bands + perturb

        # Its important to load the dataset (if asked) before transposing axes
        # to prevent from huge RAM leakage
        if load_into_RAM: self.load()

        # Define and set x-axis as the short one
        sx, sy = self.dataset.sizes['x'], self.dataset.sizes['y']
        if sx > sy:
            ds = self.dataset.rename_dims({'x': 'tmp'})
            ds = ds.rename_dims({'y': 'x'})
            self.dataset = ds.rename_dims({'tmp': 'y'})
        # Reorder axes for export to numpy array
        self.dataset = self.dataset.transpose('x', 'y', 'bands')

    def load(self):
        print('Loading hyperimage into RAM...', end='')
        self.dataset.load()
        print('   ...done!')

    ##==--------------------  Decrease dimensionality  ---------------------==##

    def downsample(self, n, axis):
        assert axis in (0, 1, 2)
        length = self.shape[axis] if axis in (0, 1) else self.nbr_bands
        ids_remove = [i for i in range(length) if i % n != 0]

        if axis == 0:
            self.dataset = self.dataset.drop_isel(x=ids_remove)
        elif axis == 1:
            self.dataset = self.dataset.drop_isel(y=ids_remove)
        else:  # axis=2
            self.dataset = self.dataset.drop_isel(bands=ids_remove)
            self._bands = np.array(self.dataset['bands'])

        if self.load_into_RAM:
            self.dataset.load()

    def trim_bands_keep(self, *kept_intervals):
        ids_remove = signop.ids_outside(self.bands, *kept_intervals)
        self.dataset = self.dataset.drop_isel(bands=ids_remove)
        self._bands = np.array(self.dataset['bands'])
        if self.load_into_RAM:
            self.dataset.load()

    def trim_bands_remove(self, *removed_intervals):
        ids_remove = signop.ids_inside(self.bands, *removed_intervals)
        self.dataset = self.dataset.drop_isel(bands=ids_remove)
        self._bands = np.array(self.dataset['bands'])
        if self.load_into_RAM:
            self.dataset.load()

    def trim_spatial_keep(self, *intervals, axis):
        assert axis in (0, 1)

        # Convert relative intervals to absolute pixel locations
        intervals = [self._spatial_rel2abs(*v, axis=axis) for v in intervals]

        ids = np.arange(self.shape[axis], dtype=int)
        ids_keep = list(set().union(*[range(v[0], v[-1] + 1) for v in intervals]))
        ids_remove = [i for i in ids if i not in ids_keep]

        if axis == self.axis_id_x:
            self.dataset = self.dataset.drop_isel(x=ids_remove)
        else:  # axis=self.axis_id_y
            self.dataset = self.dataset.drop_isel(y=ids_remove)

        if self.load_into_RAM:
            self.dataset.load()

    def trim_spatial_remove(self, *intervals, axis):
        assert axis in (0, 1)

        # Convert relative intervals to absolute pixel locations
        intervals = [self._spatial_rel2abs(*v, axis=axis) for v in intervals]

        ids_remove = list(set().union(*[range(v[0], v[-1] + 1) for v in intervals]))

        if axis == self.axis_id_x:
            self.dataset = self.dataset.drop_isel(x=ids_remove)
        else:  # axis=self.axis_id_y
            self.dataset = self.dataset.drop_isel(y=ids_remove)

        if self.load_into_RAM:
            self.dataset.load()

    def get_chunks_np(self, nbr_chunks: int, axis=1):
        """Iterate over chunks along y-axis.

        Example
        -------
        >>> for chunk in self.get_chunks_np(10):
        >>>     print(chunk.shape)  # Get numpy array

        Returns
        -------
        Iterator(np.ndarray)
            Chunks of the hyperimage.
        """
        assert axis in (self.axis_id_x, self.axis_id_y)

        ids = np.arange(self.shape[axis])
        chunks_ids = np.array_split(ids, nbr_chunks)

        da = self.dataset.to_array()
        for chunk_ids in chunks_ids:
            if axis == self.axis_id_x:
                yield da[0, chunk_ids, :, :].as_numpy()
            else:
                yield da[0, :, chunk_ids, :].as_numpy()


    ##==--------------------  Access spectra and maps  ---------------------==##

    def pixel_spectrum(self, coords: tuple[int, int]) -> np.ndarray:
        x, y = coords
        return np.array(self.dataset.mapping.isel(x=x, y=y))

    def band_map(self, band):
        band_index = self.find_closest_band(band)
        return np.array(self.dataset.mapping.isel(bands=[band_index]))[:, :, 0]

    def similarity_map(self, *signals_ref: np.array):
        # TODO: allow to choose similarity method
        signals_ref = [signop.standardization(signal) for signal in signals_ref]

        simmap = np.zeros((*self.shape, len(signals_ref)))
        for x in range(self.shape[0]):
            if x % 20 == 0: print('x=', x)
            for y in range(self.shape[1]):
                signal_xy = signop.standardization(self.pixel_spectrum((x, y)))
                for i, signal in enumerate(signals_ref):
                    simmap[x, y, i] = signop.compare_spectral_angle_mapping(
                                                              signal_xy, signal)
        if len(signals_ref) == 1:
            simmap = simmap[:, :, 0]
        return simmap

    ##==--------------------  Convert class instance  ----------------------==##

    def to_np_hyperimage(self):
        him = self.dataset.to_array().to_numpy()[0]
        return NPHyperImage(self._bands, him, self._sp_type, dtype=him.dtype)


####====------------------  Hyperimage operations  ---------------------====####

def normalize_pixels(him: np.ndarray | HyperImage,
                     upper_bound: float = 1) -> None:
    """In place scaling of individual pixel spectra to be in [0, upper_bound].

    Parameters
    ----------
    him: np.ndarray
        Hyperimage with shape (N_x, N_y, N_bands)
    upper_bound: float
        To maximum value of the spectrum such that it is comprised in
        [0, upper_bound]
    """
    sx, sy, sb = him.shape  # Initial shape
    vhim: np.view = him.view().reshape((sx*sy, sb))  # Shape (N_xy, N_bands)
    vhim = vhim.swapaxes(0, 1)                       # Shape (N_bands, N_xy)

    maxs = np.nanmax(vhim, axis=0)   # Max value along spectral axis
    vhim /= maxs                     # Normalize in [0, 1]
    vhim *= upper_bound

def normalize_global(him: np.ndarray,
                     upper_bound: float = 1) -> None:
    """In place scaling of all intensities so the maximum one is upper_bound.

    Parameters
    ----------
    him: np.ndarray
        Hyperimage with shape (N_x, N_y, N_bands)
    upper_bound: float
        To value to scale the entire map intensities in [0, upper_bound]
    """
    max_intensity = np.nanmax(him)   # Max value along spectral axis
    him /= max_intensity            # Normalize in [0, 1]
    him *= upper_bound

def standardize(him: np.ndarray) -> None:
    """In place standardization of each pixel spectrum so it has zero-mean and
    unitary standard deviation."""
    sx, sy, sb = him.shape  # Initial shape
    vhim = him.view().reshape((sx * sy, sb))  # Shape: (N_xy, N_bands)
    vhim = vhim.swapaxes(0, 1)                  # Shape: (N_bands, N_xy)

    means = np.nanmean(vhim, axis=0)  # Mean value along spectral axis
    stds = np.nanstd(vhim, axis=0)  # Std value along spectral axis
    vhim -= means
    mask = stds != 0  # To prevent from dividing by 0
    vhim[:, mask] /= stds[mask]

def standardized(him: np.ndarray) -> np.ndarray:
    """Standardization of each pixel spectum so it has zero-mean and
    unitary standard deviation.

    This method returns a standardized copy of the hyperimage.
    """
    sx, sy, sb = him.shape  # Initial shape
    him = him.copy()
    vhim = him.view().reshape((sx * sy, sb))  # Shape: (N_xy, N_bands)
    vhim = vhim.swapaxes(0, 1)                 # Shape: (N_bands, N_xy)

    means = np.nanmean(vhim, axis=0)  # Mean value along spectral axis
    stds = np.nanstd(vhim, axis=0)  # Std value along spectral axis
    vhim -= means
    mask = stds != 0  # To prevent from dividing by 0
    vhim[:, mask] /= stds[mask]

    return him

def similarity_map_sam(him: np.ndarray, signal_ref: np.array):
    """Perform similarity map using Spectral Angle Mapping (SAM) method.

    Band sampling is assumed the same between hyperimage and signal spectra.

    Parameters
    ----------
    him: np.ndarray
        Hyperimage of shape (N_x, N_y, N_bands)
    signal_ref: np.array
        Signal to compare the map with, whose length is N_bands

    Returns
    -------
    map: np.ndarray
        Similarity map with shape (N_x, N_y)
    """
    sx, sy, sb = him.shape
    assert len(signal_ref) == sb

    vhim = him.view().reshape((sx*sy, sb))

    r = signal_ref / np.linalg.norm(signal_ref)
    # Use generic array 'results' to pass it with 'out' argument.
    results = np.dot(vhim, r) / np.linalg.norm(vhim, axis=1)  # Dot products

    # Ensure valid values in arccos
    np.clip(results, a_min=-1, a_max=1, out=results)

    np.arccos(results, out=results)  # Angles
    simmap = results.view().reshape((sx, sy))

    return simmap

def similarity_map_sse(him: np.ndarray, signal_ref: np.array,
                       skip_scaling: bool = False):
    """Perform similarity map using Sum of Squared Errors (SSE) method.

    Band sampling is assumed the same between hyperimage and signal spectra.

    Parameters
    ----------
    him: np.ndarray
        Hyperimage of shape (N_x, N_y, N_bands)
    signal_ref: np.array
        Signal to compare the map with, whose length is N_bands
    skip_scaling: bool = False
        Whether the standardization step is skipped.

    Returns
    -------
    map: np.ndarray
        Similarity map with shape (N_x, N_y)
    """
    assert len(signal_ref) == him.shape[2]

    if not skip_scaling:
        signal_ref = signop.standardization(signal_ref)
        him = standardized(him)

    # Reshape
    sx, sy, sb = him.shape                   # Initial shape
    him = him.view().reshape((sx * sy, sb))  # Shape: (N_xy, N_bands)

    errors = him - signal_ref
    squared_errors = errors ** 2
    sum_squared_errors = np.sum(squared_errors, axis=1)
    simmap = sum_squared_errors.reshape((sx, sy))

    return simmap
