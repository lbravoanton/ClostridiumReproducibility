from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import norm
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
from maldi_nn.utils import topf
import matplotlib.pyplot as plt
import h5torch
import torch


class SpectrumObject:
    """Base Spectrum Object class

    Can be instantiated directly with 1-D np.arrays for mz and intensity.
    Alternatively, can be read from csv files or from bruker output data.
    Reading from Bruker data is based on the code in https://github.com/sgibb/readBrukerFlexData

    Parameters
    ----------
    mz : 1-D np.array, optional
        mz values, by default None
    intensity : 1-D np.array, optional
        intensity values, by default None
    """

    def __init__(self, mz=None, intensity=None):
        self.mz = mz
        self.intensity = intensity
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        if self.mz is not None:
            return self.mz.shape[0]
        else:
            return 0

    def plot(self, as_peaks=False):
        """Plot a spectrum via matplotlib

        Parameters
        ----------
        as_peaks : bool, optional
            draw points in the spectrum as individualpeaks, instead of connecting the points in the spectrum, by default False
        """
        if as_peaks:
            mz_plot = np.stack([self.mz - 1, self.mz, self.mz + 1]).T.reshape(-1)
            int_plot = np.stack(
                [
                    np.zeros_like(self.intensity),
                    self.intensity,
                    np.zeros_like(self.intensity),
                ]
            ).T.reshape(-1)
        else:
            mz_plot, int_plot = self.mz, self.intensity
        plt.plot(mz_plot, int_plot)

    def __repr__(self):
        string_ = np.array2string(
            np.stack([self.mz, self.intensity]), precision=5, threshold=10, edgeitems=2
        )
        mz_string, int_string = string_.split("\n")
        mz_string = mz_string[1:]
        int_string = int_string[1:-1]
        return "SpectrumObject([\n\tmz  = %s,\n\tint = %s\n])" % (mz_string, int_string)

    @staticmethod
    def tof2mass(ML1, ML2, ML3, TOF):
        A = ML3
        B = np.sqrt(1e12 / ML1)
        C = ML2 - TOF

        if A == 0:
            return (C * C) / (B * B)
        else:
            return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    @classmethod
    def from_bruker(cls, acqu_file, fid_file):
        """Read a spectrum from Bruker's format

        Parameters
        ----------
        acqu_file : str
            "acqu" file bruker folder
        fid_file : str
            "fid" file in bruker folder

        Returns
        -------
        SpectrumObject
        """
        with open(acqu_file, "rb") as f:
            lines = [line.decode("utf-8", errors="replace").rstrip() for line in f]
        for l in lines:
            if l.startswith("##$TD"):
                TD = int(l.split("= ")[1])
            if l.startswith("##$DELAY"):
                DELAY = int(l.split("= ")[1])
            if l.startswith("##$DW"):
                DW = float(l.split("= ")[1])
            if l.startswith("##$ML1"):
                ML1 = float(l.split("= ")[1])
            if l.startswith("##$ML2"):
                ML2 = float(l.split("= ")[1])
            if l.startswith("##$ML3"):
                ML3 = float(l.split("= ")[1])
            if l.startswith("##$BYTORDA"):
                BYTORDA = int(l.split("= ")[1])
            if l.startswith("##$NTBCal"):
                NTBCal = l.split("= ")[1]

        intensity = np.fromfile(fid_file, dtype={0: "<i", 1: ">i"}[BYTORDA])

        if len(intensity) < TD:
            TD = len(intensity)
        TOF = DELAY + np.arange(TD) * DW

        mass = cls.tof2mass(ML1, ML2, ML3, TOF)

        intensity[intensity < 0] = 0

        return cls(mz=mass, intensity=intensity)

    @classmethod
    def from_tsv(cls, file, sep=","):
        """Read a spectrum from txt

        Parameters
        ----------
        file : str
            path to csv file
        sep : str, optional
            separator in the file, by default " "

        Returns
        -------
        SpectrumObject
        """
        s = pd.read_table(
            file, sep=sep, index_col=None, comment="#", header=None
        ).values
        mz = np.int32(s[:, 0])
        intensity = np.int32(s[:, 1])
        return cls(mz=mz, intensity=intensity)

    def torch(self):
        """Converts spectrum to dict of tensors"""
        return {"mz": torch.tensor(self.mz), "intensity": torch.tensor(self.intensity)}


class Binner:
    """Pre-processing function for binning spectra in equal-width bins.

    Parameters
    ----------
    start : int, optional
        start of the binning range, by default 2000
    stop : int, optional
        end of the binning range, by default 20000
    step : int, optional
        width of every bin, by default 3
    aggregation : str, optional
        how to aggregate intensity values in each bin.
        Is passed to the statistic argument of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        Can take any argument that the statistic argument also takes, by default "sum"
    """

    def __init__(self, start=2000, stop=20000, step=3, aggregation="sum"):
        self.bins = np.arange(start, stop + 1e-8, step)
        self.mz_bins = self.bins[:-1] + step / 2
        self.agg = aggregation

    def __call__(self, SpectrumObj):
        if self.agg == "sum":
            bins, _ = np.histogram(
                SpectrumObj.mz, self.bins, weights=SpectrumObj.intensity
            )
        else:
            bins = binned_statistic(
                SpectrumObj.mz,
                SpectrumObj.intensity,
                bins=self.bins,
                statistic=self.agg,
            ).statistic
            bins = np.nan_to_num(bins)

        s = SpectrumObject(intensity=bins, mz=self.mz_bins)
        return s

class BinnerLog:
    """Pre-processing function for binning spectra in logaritmic width bins.

    Parameters
    ----------
    start : int, optional
        start of the binning range, by default 2000
    stop : int, optional
        end of the binning range, by default 20000
    num_bins : int, optional
        total number of bins, by default 100
    aggregation : str, optional
        how to aggregate intensity values in each bin.
        Is passed to the statistic argument of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        Can take any argument that the statistic argument also takes, by default "sum"
    """

    def __init__(self, start=2000, stop=20000, num_bins=100, aggregation="sum"):
        self.start = start
        self.stop = stop
        self.num_bins = num_bins
        self.agg = aggregation
    def __call__(self, SpectrumObj):
        bin_edges = np.logspace(np.log10(self.start), np.log10(self.stop), num=self.num_bins+1)

        # Calculate bin centers
        mz_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

        if self.agg == "sum":
            bins, _ = np.histogram(
                SpectrumObj.mz, bin_edges, weights=SpectrumObj.intensity
            )
        else:
            bins = binned_statistic(
                SpectrumObj.mz,
                SpectrumObj.intensity,
                bins=bin_edges,
                statistic=self.agg,
            ).statistic
            bins = np.nan_to_num(bins)

        s = SpectrumObject(intensity=bins, mz=mz_bins)
        return s

'''class BinnerLog:
    """Pre-processing function for binning spectra in logaritmic width bins.

    Parameters
    ----------
    start : int, optional
        start of the binning range, by default 2000
    stop : int, optional
        end of the binning range, by default 20000
    num_bins : int, optional
        total number of bins, by default 100
    aggregation : str, optional
        how to aggregate intensity values in each bin.
        Is passed to the statistic argument of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        Can take any argument that the statistic argument also takes, by default "sum"
    """

    def __init__(self, start=2000, stop=20000, num_bins=100, aggregation="sum"):
        self.start = np.log10(start)
        self.stop = np.log10(stop)
        self.num_bins = num_bins
        step = (self.stop - self.start) / (self.num_bins+1)
        self.bins = np.arange(self.start, self.stop, step)
        self.mz_bins = self.bins[:-1] + step / 2
        self.agg = aggregation

    def __call__(self, SpectrumObj):
        if self.agg == "sum":
            bins, _ = np.histogram(
                np.log10(SpectrumObj.mz), self.bins, weights=SpectrumObj.intensity
            )
        else:
            bins = binned_statistic(
                SpectrumObj.mz,
                SpectrumObj.intensity,
                bins=self.bins,
                statistic=self.agg,
            ).statistic
            bins = np.nan_to_num(bins)

        s = SpectrumObject(intensity=bins, mz=self.mz_bins)
        return s'''

class BinnerDynamic:
    """Pre-processing function for binning spectra in logaritmic width bins.

    Parameters
    ----------
    bin_size: list with the size of the bins by range
    mass_ranges: list with the mass ranges associated with the bins sizes in the above list
    aggregation : str, optional
        how to aggregate intensity values in each bin.
        Is passed to the statistic argument of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        Can take any argument that the statistic argument also takes, by default "sum"
    """

    def __init__(self, bin_size=[10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20,  30, 30, 30, 30, 50, 50, 50,50, 70, 70, 70, 70, 100, 100,  100, 100, 125,125, 125, 125, 150,150, 150, 150, 175,175, 175, 175, 200, 200, 200,  200, 250, 250,  250, 250, 275, 275,275,275,
            300, 300, 300, 300, 325, 325,325,325, 350, 350, 350, 400, 400, 400, 450, 450, 450, 500, 500], mass_ranges=[2000,2050, 2100, 2250,2300, 2500,2750, 3000, 3250, 3500,3750, 4000, 4250, 4500,4750, 5000,5250, 5500,5750, 6000,6250, 6500, 7000, 7250, 7500, 8000,8250,
                                                                                     8500, 8750, 9000, 9250, 9500,9750, 10000,10500, 11000,11500, 12000,1250,12500, 12750,13000,13250, 13500, 13750, 14000, 14250, 14500, 14750,
                                                                             15000, 15250, 15500, 15750, 16000, 16250, 16500, 16750, 17000, 17250, 17500, 17750,  18000, 18250, 18500, 18750, 19000, 19250, 19500, 19750], aggregation="sum"):
        self.bin_size = bin_size
        self.mass_ranges = np.array(mass_ranges)
        self.agg = aggregation

    def bin_dataset(self, obj, bin_size=3):
        bins = []

        for i in range(0, len(obj), bin_size):      
            
            bin_sum = np.sum(obj[i:i+bin_size])
            bins.append(bin_sum)


        return bins

    def dynamic_bin_dataset(self, SpectrumObj, bin_sizes, mass_ranges):
        bins = []
        intensities = np.array(SpectrumObj.intensity)
        mz = np.array(SpectrumObj.mz)
       
        for i, (start_mass, end_mass) in enumerate(zip(mass_ranges[:-1], mass_ranges[1:])):

            start_index = np.where(np.round(mz)>=start_mass)[0][0]
            end_index = np.where(np.round(mz)>end_mass)[0][0]
            sub = intensities[start_index:end_index]
            bins.append(np.sum(self.bin_dataset(sub, bin_size=bin_sizes[i])))

        return bins
    def __call__(self, SpectrumObj):

        mz_bins = (self.mass_ranges[:-1] + self.mass_ranges[1:]) / 2
        bins =  self.dynamic_bin_dataset(SpectrumObj, self.bin_size, self.mass_ranges)  
        s = SpectrumObject(intensity=np.array(bins), mz=np.array(mz_bins))
        return s

class Normalizer:
    """Pre-processing function for normalizing the intensity of a spectrum.
    Commonly referred to as total ion current (TIC) calibration.

    Parameters
    ----------
    sum : int, optional
        Make the total intensity of the spectrum equal to this amount, by default 1
    """

    def __init__(self, sum=1):
        self.sum = sum

    def __call__(self, SpectrumObj):
        s = SpectrumObject()

        s = SpectrumObject(
            intensity=SpectrumObj.intensity / SpectrumObj.intensity.sum() * self.sum,
            mz=SpectrumObj.mz,
        )
        return s


class Trimmer:
    """Pre-processing function for trimming ends of a spectrum.
    This can be used to remove inaccurate measurements.

    Parameters
    ----------
    min : int, optional
        remove all measurements with mz's lower than this value, by default 2000
    max : int, optional
        remove all measurements with mz's higher than this value, by default 20000
    """

    def __init__(self, min=2000, max=20000):
        self.range = [min, max]

    def __call__(self, SpectrumObj):
        indices = (self.range[0] < SpectrumObj.mz) & (SpectrumObj.mz < self.range[1])

        s = SpectrumObject(
            intensity=SpectrumObj.intensity[indices], mz=SpectrumObj.mz[indices]
        )
        return s


class VarStabilizer:
    """Pre-processing function for manipulating intensities.
    Commonly performed to stabilize their variance.

    Parameters
    ----------
    method : str, optional
        function to apply to intensities.
        can be either "sqrt", "log", "log2" or "log10", by default "sqrt"
    """

    def __init__(self, method="sqrt"):
        methods = {"sqrt": np.sqrt, "log": np.log, "log2": np.log2, "log10": np.log10}
        self.fun = methods[method]

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=self.fun(SpectrumObj.intensity), mz=SpectrumObj.mz)
        return s


class BaselineCorrecter:
    """Pre-processing function for baseline correction (also referred to as background removal).

    Support SNIP, ALS and ArPLS.
    Some of the code is based on https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    Parameters
    ----------
    method : str, optional
        Which method to use
        either "SNIP", "ArPLS" or "ALS", by default None
    als_lam : float, optional
        lambda value for ALS and ArPLS, by default 1e8
    als_p : float, optional
        p value for ALS and ArPLS, by default 0.01
    als_max_iter : int, optional
        max iterations for ALS and ArPLS, by default 10
    als_tol : float, optional
        stopping tolerance for ALS and ArPLS, by default 1e-6
    snip_n_iter : int, optional
        iterations of SNIP, by default 10
    """

    def __init__(
        self,
        method=None,
        als_lam=1e8,
        als_p=0.01,
        als_max_iter=10,
        als_tol=1e-6,
        snip_n_iter=10,
    ):
        self.method = method
        self.lam = als_lam
        self.p = als_p
        self.max_iter = als_max_iter
        self.tol = als_tol
        self.n_iter = snip_n_iter

    def __call__(self, SpectrumObj):
        if "LS" in self.method:
            baseline = self.als(
                SpectrumObj.intensity,
                method=self.method,
                lam=self.lam,
                p=self.p,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        elif self.method == "SNIP":
            baseline = self.snip(SpectrumObj.intensity, self.n_iter)

        s = SpectrumObject(
            intensity=SpectrumObj.intensity - baseline, mz=SpectrumObj.mz
        )
        return s

    def als(self, y, method="ArPLS", lam=1e8, p=0.01, max_iter=10, tol=1e-6):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(
            D.transpose()
        )  # Precompute this term since it does not depend on `w`

        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)

        crit = 1
        count = 0
        while crit > tol:
            z = sparse.linalg.spsolve(W + D, w * y)

            if method == "ALS":
                w_new = p * (y > z) + (1 - p) * (y < z)
            elif method == "ArPLS":
                d = y - z
                dn = d[d < 0]
                m = np.mean(dn)
                s = np.std(dn)
                w_new = 1 / (1 + np.exp(np.minimum(2 * (d - (2 * s - m)) / s, 70)))

            crit = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)
            count += 1
            if count > max_iter:
                break
        return z

    def snip(self, y, n_iter):
        y_prepr = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
        for i in range(1, n_iter + 1):
            rolled = np.pad(y_prepr, (i, i), mode="edge")
            new = np.minimum(
                y_prepr, (np.roll(rolled, i) + np.roll(rolled, -i))[i:-i] / 2
            )
            y_prepr = new
        return (np.exp(np.exp(y_prepr) - 1) - 1) ** 2 - 1


class Smoother:
    """Pre-processing function for smoothing. Uses Savitzky-Golay filter.

    Parameters
    ----------
    halfwindow : int, optional
        halfwindow of savgol_filter, by default 10
    polyorder : int, optional
        polyorder of savgol_filter, by default 3
    """

    def __init__(self, halfwindow=10, polyorder=3):
        self.window = halfwindow * 2 + 1
        self.poly = polyorder

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=np.maximum(
                savgol_filter(SpectrumObj.intensity, self.window, self.poly), 0
            ),
            mz=SpectrumObj.mz,
        )
        return s


class PersistenceTransformer:
    """Pre-processing function for Peak Detection.
    Uses the Persistance Transformation first outlined in https://doi.org/10.1093/bioinformatics/btaa429
    Underlying code is from https://github.com/BorgwardtLab/Topf

    Parameters
    ----------
    extract_nonzero : bool, optional
        whether to extract detected peaks or to keep zeros in, by default False
    """

    def __init__(self, extract_nonzero=False):
        self.filter = extract_nonzero

    def __call__(self, SpectrumObj):
        a = np.stack([SpectrumObj.mz, SpectrumObj.intensity]).T
        b = topf.PersistenceTransformer().fit_transform(a)

        s = SpectrumObject()
        if self.filter:
            peaks = b[:, 1] != 0
            s = SpectrumObject(intensity=b[peaks, 1], mz=b[peaks, 0])
        else:
            s = SpectrumObject(intensity=b[:, 1], mz=b[:, 0])
        return s


class PeakFilter:
    """Pre-processing function for filtering peaks.

    Filters in two ways: absolute number of peaks and height.

    Parameters
    ----------
    max_number : int, optional
        Maximum number of peaks to keep. Prioritizes peaks to keep by height.
        by default None, for no filtering
    min_intensity : float, optional
        Min intensity of peaks to keep, by default None, for no filtering
    """

    def __init__(self, max_number=None, min_intensity=None):
        self.max_number = max_number
        self.min_intensity = min_intensity

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=SpectrumObj.intensity, mz=SpectrumObj.mz)

        if self.max_number is not None:
            indices = np.argsort(-s.intensity, kind="stable")
            take = np.sort(indices[: self.max_number])

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        if self.min_intensity is not None:
            take = s.intensity >= self.min_intensity

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        return s


class RandomPeakShifter:
    """Pre-processing function for adding random (gaussian) noise to the mz values of peaks.

    Parameters
    ----------
    std : float, optional
        stdev of the random noise to add, by default 1
    """

    def __init__(self, std=1.0):
        self.std = std

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.normal(scale=self.std, size=SpectrumObj.mz.shape),
        )
        return s


class UniformPeakShifter:
    """Pre-processing function for adding uniform noise to the mz values of peaks.

    Parameters
    ----------
    range : float, optional
        let each peak shift by maximum this value, by default 1.5
    """

    def __init__(self, range=1.5):
        self.range = range

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.uniform(
                low=-self.range, high=self.range, size=SpectrumObj.mz.shape
            ),
        )
        return s


class Binarizer:
    """Pre-processing function for binarizing intensity values of peaks.

    Parameters
    ----------
    threshold : float
        Threshold for the intensities to become 1 or 0.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=(SpectrumObj.intensity > self.threshold).astype(
                SpectrumObj.intensity.dtype
            ),
            mz=SpectrumObj.mz,
        )
        return s


class SequentialPreprocessor:
    """Chain multiple preprocessors so that a pre-processing pipeline can be called with one line.

    Example:
    ```python
    preprocessor = SequentialPreprocessor(
        VarStabilizer(),
        Smoother(),
        BaselineCorrecter(method="SNIP"),
        Normalizer(),
        Binner()
    )
    preprocessed_spectrum = preprocessor(spectrum)
    ```
    """

    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, SpectrumObj):
        for step in self.preprocessors:
            SpectrumObj = step(SpectrumObj)
        return SpectrumObj



class ScaleNormalizer:
    """
    Normalizes a set of spectra such that their scales are not too
    small (greater than one).
    """

    def _calculate_min_nonzero_intensity(self, spectra):
        intensities = np.concatenate(
            [
                s.intensity[s.intensity != 0] for s in spectra
            ],
            axis=0
        )
        return np.min(intensities)

    def _normalize_spectrum(self, spectrum, min):
        
        scaling = 1.0 / min
        return spectrum * np.array([1, scaling])[np.newaxis,:]


    def transform(self, X):
        self.min_nonzero_intensity = self._calculate_min_nonzero_intensity(X)
        return [
            self._normalize_spectrum(spectrum, self.min_nonzero_intensity ) for spectrum in X
        ]