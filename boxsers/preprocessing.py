"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides functions to preprocess vibrational spectra. These features
improve spectrum quality and can improve performance for machine learning applications
"""
import numpy as np
from scipy import interpolate, sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, medfilt
from sklearn.preprocessing import normalize


def als_baseline_cor(sp, lam=1e4, p=0.001, niter=10, return_baseline=False):
    """
    Subtracts the baseline signal from the spectrum(s) using Asymmetric Least Squares estimation.
    (Updated April 2020: improved computing speed)

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lam : integer or float,  default = 1e4
            ALS 2nd derivative constraint that defines the smoothing degree of the baseline correction.

        p : int or float, default=0.001
            ALS positive residue weighting that defines the asymmetry of the baseline correction.

        niter : int, default=10
            Maximum number of iterations to optimize the baseline.

        return_baseline: Boolean, default=False
            If True, the function also returns the baseline array.

    Returns:
        (array) Baseline substracted spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra
                and = (n_pixels,) for a single spectrum.

        (array)(OPTIONAL) Baseline signal(s). Array shape = (n_spectra, n_pixels) for multiple spectra
                and = (n_pixels,) for a single spectrum.
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    # initialization and space allocation
    baseline = np.zeros(sp.shape)  # baseline signal array
    sp_length = sp.shape[1]  # length of a spectrum
    diag = sparse.diags([1, -2, 1], [0, -1, -2], shape=(sp_length, sp_length - 2))
    diag = lam * diag.dot(diag.transpose())
    w = np.ones(sp_length)
    w_matrix = sparse.spdiags(w, 0, sp_length, sp_length)

    for n in range(0, len(sp)):
        for i in range(niter):
            w_matrix.setdiag(w)
            z = w_matrix + diag
            baseline[n] = spsolve(z, w * sp[n])
            w = p * (sp[n] > baseline[n]) + (1 - p) * (sp[n] < baseline[n])  # w is updated according to baseline

    if return_baseline:
        return sp-baseline, baseline

    return sp-baseline


def cosmic_filter(sp, ks=3):
    """
    Apply a median filter to the spectrum(s) to remove cosmic rays.

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
             for a single spectrum.

        ks : positive odd integer, default = 3
            Size of the median filter window in pixel.

    Returns:
        (array) Filtered spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.
    """
    sp = np.array(sp, ndmin=2)  # sp is forced to be a two-dimensional array
    ks_1d = (1, ks)
    sp_med = medfilt(sp, ks_1d)  # from scipy
    return sp_med


def spectral_normalization(sp, norm='l2'):
    """ Normalizes the spectrum(s) using one of the available norms in this function.

    Notes:
        The snv norm corresponds to 'Standard Normal Variate' method.

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
             for a single spectrum.

        norm : {'l2', 'l1', 'max', 'maxmin', 'snv'}, default = 'max'
            Procedure used to normalize/scale each spectrum.
                - 'l2': The sum of the squared values of the spectrum is equal to 1.
                - 'l1': The sum of the absolute values of the spectrum is equal to 1.
                - 'max': The maximum value of the spectrum is equal to 1.
                - 'minmax': The values of the spectrum are scaled between 0 and 1.
                - 'snv': The mean and the standard deviation of the spectrum are respectively equal to 0 and 1.

    Returns:
        (array) Normalized spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)

    # max, min, mean, std calculation for each spectrum
    sp_max = np.max(sp, axis=1, keepdims=True)
    sp_min = np.min(sp, axis=1, keepdims=True)
    sp_mean = np.mean(sp, axis=1, keepdims=True)
    sp_std = np.std(sp, axis=1, keepdims=True)

    # normalization operations
    if norm in {'l2', 'l1', 'max'}:
        return normalize(sp, norm=norm)  # from sklearn
    if norm == 'minmax':
        return (sp-sp_min)/(sp_max-sp_min)
    if norm == 'snv':
        return (sp-sp_mean)/sp_std

    raise ValueError(norm, 'is not among the following valid choices:\'l2\', \'l1\', \'max\', \'minmax\', \'snv\'')


def savgol_smoothing(sp, window_length=9, p=3, degree=0):
    """
    Smoothes the spectrum(s) using a Savitzky-Golay polynomial filter.

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        window_length : odd integer, default=9
            Savitzky-Golay filters moving window length. Must be less than or equal to the length of the spectra.

        p : int, default=3
            Polynomial order used to fit the spectra. Must be less than window_length.

        degree: Positive integer, default=0
            Savitzky-Golay derivative order.

    Returns:
        (array) Smoothed spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.
    """
    #  Scipy's Savitzky-Golay filter is used
    sp_svg = savgol_filter(sp, window_length, polyorder=p, deriv=degree)  # from Scipy
    return sp_svg


def spectral_cut(sp, wn, wn_start, wn_end, sub_mode='zero'):
    """
    Subtracts or sets to zero a delimited spectral region of the spectrum(s)

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
             for a single spectrum.

        wn : array
            X-axis(wavenumber, wavelenght, Raman shift, etc.) used for the spectra. Array shape = (n_pixels, ).

        wn_start : Int or float
            Starting point (same unit as wn) of the subtracted spectral region.

        wn_end : Int or float
            Ending point (same unit as wn) of the subtracted spectral region.

        sub_mode : {'zero', 'remove'}, default='zero'
            Determines how the subtracted part of the spectrum is handled.
                - 'zero': The subtracted part of the spectrum is set to zero.
                - 'remove': The subtracted part of the spectrum is removed and the remaining parts are joined together.

    Returns:
        (array) Cutted spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.

        (array) New x-axis( if sub_mode = 'removed') or the intial one(if sub_mode = 'zero').
                Array shape = (n_pixels, ).
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    # conversion to indexes
    i_start = (np.abs(wn - wn_start)).argmin()
    i_end = (np.abs(wn - wn_end)).argmin()
    if sub_mode == 'zero':
        # preserved part on the left side of the spectral cut
        sp[:, i_start:i_end] = 0
        return sp, wn

    elif sub_mode == 'remove':
        # preserved part on the left side of the spectral cut
        sp_l = sp[:, 0:i_start]
        wn_l = wn[0:i_start]
        # preserved part on the right side of the spectral cut
        sp_r = sp[:, i_end:]
        wn_r = wn[i_end:]
        # remaining parts are joined together
        sp_cut = np.concatenate([sp_l, sp_r], axis=1)
        wn_cut = np.concatenate([wn_l, wn_r])
        return sp_cut, wn_cut

    raise ValueError('invalid sub_mode among \'zero\' and \'remove\'')


def spline_interpolation(sp, wn, new_wn, degree=1, same_w=False):
    """
    Performs a one-dimensional interpolation spline on the spectra to reproduce them with a new x-axis.

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        wn : array or list of array
            X-axis(wavenumber, wavelenght, Raman shift, etc.) used for the spectra. Array shape = (n_pixels,).

        new_wn : array
            New x-axis(wavenumber, wavelenght, Raman shift, etc.) used for the spectra, array shape = (n_pixels,).

        degree : int, default=1
            Spline interpolation degree.

        same_w : Boolean, default=True
            Sets to True if the same x-axis is used for all spectra.

    Returns:
        (array) Interpolated spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.

        (list of integer values) List that contains the erroneous spectrum indexes.
    """
    # initializing an empty array with shape = (number of spectra x new wavenumber length)
    new_sp = np.empty((sp.shape[0], len(new_wn)))
    # initializinz an empty list to stock indexes where the errors occurred
    error_ind = []
    value_error_detected_event = 0  # no errors initially detected

    if same_w:  # the same wavenumbers are used for all spectra
        # wavenumber are replicated multiple times in a array for the next opperations
        wn = np.tile(wn, (sp.shape[0], 1))
    elif wn.shape[0] != sp.shape[0]:
        raise ValueError('spectra and wavenumber/wavelength sizes do not fit')

    for i in range(sp.shape[0]):
        try:
            s = interpolate.InterpolatedUnivariateSpline(wn[i], sp[i], k=degree, ext=1)
            new_sp[i] = s(new_wn)
        except ValueError:
            # Display a message at the end of the function if an error has occurred.
            value_error_detected_event = 1
            # Saves the indexes where the errors occurred
            error_ind.append(i)

    if value_error_detected_event == 1:
        new_sp = np.delete(new_sp, error_ind, 0)
        print('Some spectra raised errors and were removed')
        print('Erroneous spectrum indexes are identified by the second output of this function.')
        print('Delete the corresponding label with this command: new_label = np.delete(label, erroneous_index)')
    return new_sp, error_ind


if __name__ == "__main__":
    help(__name__)
