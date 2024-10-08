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
from scipy.spatial import ConvexHull
from sklearn.preprocessing import normalize


def _find_raman_indices(wn, raman_region):
    if isinstance(raman_region, (list, tuple)) and len(raman_region) == 2:
        wn_start, wn_end = raman_region
        mask = (wn >= wn_start) & (wn <= wn_end)
        indices = np.where(mask)[0]
        if indices.size > 0:
            return indices
        else:
            return "No indices were found in the given range."
    elif isinstance(raman_region, (int, float)):
        center = (np.abs(wn - raman_region)).argmin()
        print('center', center)
        return [center-1, center, center+1]
    else:
        raise TypeError('Invalid raman_region type, valid choices: int, float, list, tuple')


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
            Suggested lam range = [1e2, 1e8]

        p : int or float, default=0.001
            ALS positive residue weighting that defines the asymmetry of the baseline correction. Negative
            weighting is equal to (1-p). Suggested p range = [0.001, 0.1]

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


def rubberband_baseline_cor(sp, return_baseline=False):
    """
    Notes:
        - code mainly coming from:
        https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data
        - Compared with the als_baseline_cor method, this technique is faster
          to run, but has no parameters to adjust to optimize the correction.

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        return_baseline : Boolean, default=False
            If True, the function also returns the baseline array.

    Returns:
        (array) Baseline substracted spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra
                and = (n_pixels,) for a single spectrum.

        (array)(OPTIONAL) Baseline signal(s). Array shape = (n_spectra, n_pixels) for multiple spectra
                and = (n_pixels,) for a single spectrum.
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    n_spectra, sp_length = sp.shape  # number of spectra, spectrum length

    # initialization and space allocation
    baseline = np.zeros(sp.shape)  # baseline signal array
    indexes = np.arange(sp_length)

    for i in range(n_spectra):
        # find the vertices of the spectrum's convex hull
        convex_v = ConvexHull(np.array(list(zip(indexes, sp[i])))).vertices
        # rolls the vertex indices of the convex hull until they start from the lowest element of the array
        convex_v = np.roll(convex_v, -convex_v.argmin())
        # keeps only the ascending part of the vertexes determined above
        convex_v = convex_v[:(convex_v.argmax()+1)]

        # baseline is generated by linear interpolation between the remaining vertices
        baseline[i] = np.interp(indexes, indexes[convex_v], sp[i, convex_v])

    if return_baseline:
        # returns the baseline signal if requested
        return sp - baseline, baseline
    return sp - baseline


def rollingball_baseline_cor(sp, window=40, smoothing_window=None, return_baseline=False):
    """
        Notes:
            - Kneen, M. A.; Annegarn, H. J. Nuclear Instruments and Methods in Physics Research Section B: Beam
              Interactions with Materials and Atoms 1996, 109–110, 209–213.

        Parameters:
            sp : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.

            window : int, default=50
                Half-width the window, or radius, of the rolling ball used to calculate the baseline.

            smoothing_window : int, default=None,
                Half-width of the window used to smooth the baseline in the last step. If None, the rolling ball
                window value is used as smoothing_window.

            return_baseline : Boolean, default=False
                If True, the function also returns the baseline array.

        Returns:
            (array) Baseline substracted spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra
                    and = (n_pixels,) for a single spectrum.

            (array)(OPTIONAL) Baseline signal(s). Array shape = (n_spectra, n_pixels) for multiple spectra
                    and = (n_pixels,) for a single spectrum.
        """
    if smoothing_window is None:
        # if smoothing_window is None, the rolling ball window value is used as smoothing_window
        smoothing_window = window

    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    n_spectra, sp_length = sp.shape  # number of spectra, spectrum length

    # initialization and space allocation
    baseline = np.zeros_like(sp)
    loc_minima = np.zeros_like(sp)
    loc_maxima = np.zeros_like(sp)

    for i in np.arange(sp_length):
        i_start = max(0, i - window)  # first index of the rolling ball, 0 if i-window < 0
        i_end = min(sp_length, i + window + 1)  # last index of the rolling ball, sp_length if i+window+1 > sp_length
        loc_minima[:, i] = np.min(sp[:, i_start:i_end], axis=1)  # measures local minima
    for i in np.arange(sp_length):
        i_start = max(0, i - window)
        i_end = min(sp_length, i + window + 1)
        loc_maxima[:, i] = np.max(loc_minima[:, i_start:i_end], axis=1)  # measures local maxima from minima
    for i in np.arange(sp_length):
        i_start = max(0, i - smoothing_window)
        i_end = min(sp_length, i + smoothing_window + 1)
        baseline[:, i] = np.mean(loc_maxima[:, i_start:i_end], axis=1)  # Average smoothing of local maxima

    if return_baseline:
        # returns the baseline signal if requested
        return sp - baseline, baseline
    return sp - baseline


def cosmic_filter(sp, width=3, threshold=11.0):
    """
    Suppresses cosmic rays using a sliding window that compares each portion of each spectrum with the other spectra.

    Notes:
        - Does not work for an individual spectrum, since this function relies on other spectra
           to determine the presence of a cosmic peak in a spectrum.
        - The cosmic peak is replaced by the linear interpolation of the two closest values of the window.

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels). Does not work for an individual
             spectrum.

        width : positive odd integer, default = 7
            Size(in pixel) of the moving window.

        threshold : positive float, default = 11.0
            Threshold used to detect cosmic peaks. This threshold is multiplied by the standard deviation
             and compare to the median of the moving window

    Returns:
        (array) Filtered spectrum(s). Array shape = (n_spectra, n_pixels).
    """
    half_width = width // 2  # integer division
    # To avoid modifying the original spectra
    sp = sp.copy()
    filtered_spectrum = sp.copy()

    if sp.ndim == 1 or (sp.ndim == 2 and sp.shape[0] == 1):
        raise ValueError('This function does not work for an individual spectrum, since it'
                         ' relies on other spectra to determine the presence of a cosmic peak'
                         ' in a spectrum.')

    for i in range(half_width, sp.shape[1] - half_width):  # window slides over the spectral axis.
        window = sp[:, i - half_width:i + half_width + 1]
        median_value = np.median(window)  # median value of the window for the entire sp dataset
        std_value = np.std(window)  # std value of the window for the entire sp dataset

        abs_diff = np.abs(sp[:, i] - median_value)
        # Returns a mask with True values where cosmic peaks are detected
        cosmspike_mask = abs_diff > threshold * std_value

        for n in range(sp.shape[0]):  # loop over the spectra
            if cosmspike_mask[n]:  # a cosmic peak was detected
                # Following max and min function are used for spectrum edge cases
                left_index = max(i - half_width - 1, 0)
                right_index = min(i + half_width + 1, sp.shape[1] - 1)
                left_value = sp[n, left_index]
                right_value = sp[n, right_index]
                # Cosmic peak detected are removed by linear interpolation
                filtered_spectrum[n, i] = np.interp(i, [left_index, right_index], [left_value, right_value])
    return filtered_spectrum


def spectral_normalization(sp, norm='max', wn=None, raman_region=None):
    """ Normalizes the spectrum(s) using one of the available norms in this function.

    Notes:
        The snv norm corresponds to 'Standard Normal Variate' method.

    Parameters:
        sp : array
            Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        norm : {'l2', 'l1', 'max', 'maxmin', 'snv'}, default = 'max'
            Procedure used to normalize/scale each spectrum.
                - 'l2': The sqrt(sum(x^2)) of the selected spectrum region is equal to 1.
                - 'l1': The sum of the absolute values of the selected spectrum region is equal to 1.
                - 'max': The maximum value of the selected spectrum region is equal to 1.
                - 'minmax': The values of the spectrum are scaled between 0 and 1.
                - 'snv': The mean and the standard deviation of the spectrum are respectively equal to 0 and 1.

        wn :  array or list, default=None
            X-axis(wavenumber, wavelenght, Raman shift, etc.), array shape = (n_pixels, ). Considered only if
            raman_region is not None.

        raman_region : int, float, tuple or list, default=None
            Raman band (int or float) or spectral region ([start, end]) position, in wn units, used to
            calculate the norm value. If None or if norm={'maxmin' or 'snv'}, the whole spectral range is used.

    Returns:
        (array) Normalized spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)

    # definition of the raman region or band used to calculate the norm
    if raman_region is not None:
        sp_r = sp[:, _find_raman_indices(wn, raman_region)]
    else:
        sp_r = sp.copy()  # the whole spectral range is used to calculate the norm

    # max, min, mean, std calculation for each spectrum
    sp_max = np.max(sp, axis=1, keepdims=True)
    sp_min = np.min(sp, axis=1, keepdims=True)
    sp_mean = np.mean(sp, axis=1, keepdims=True)
    sp_std = np.std(sp, axis=1, keepdims=True)

    # normalization operations:
    if norm == 'max':
        return sp/np.max(sp_r, axis=1, keepdims=True)
    if norm == 'l1':
        return sp/np.sum(np.abs(sp_r), axis=1, keepdims=True)
    if norm == 'l2':
        return sp/np.sqrt(np.sum(sp_r**2, axis=1, keepdims=True))
    # normalization & scaling operations (applied only on the whole spectrum):
    if norm == 'minmax':
        return (sp-sp_min)/(sp_max-sp_min)
    if norm == 'snv':
        return (sp-sp_mean)/sp_std
    else:
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
            Polynomial order used to fit the spectra. Must be less than window_length. Suggested value = 2 or 3

        degree: Positive integer, default=0
            Savitzky-Golay derivative order.

    Returns:
        (array) Smoothed spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    #  Scipy's Savitzky-Golay filter is used
    sp_svg = savgol_filter(sp, window_length, polyorder=p, deriv=degree)  # from Scipy
    return sp_svg


def spectral_cut(sp, wn, wn_start, wn_end, sub_mode='zero', keep_it=False, reduce_gap=False):
    """
    Subtracts or sets to zero a spectral region of the spectrum(s) bounded by [wn_start, wn_end]. Can also
    be used to keep only the delimited part of the spectrum.

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
                - 'remove': The subtracted part of the spectrum is removed.

        keep_it : boolean, default=False
            If True, keeps the specified region instead and subtracts the rest.

        reduce_gap : boolean, default=False
            If True, reduces the gap to zero between the left and right parts to minimize discontinuity.

    Returns:
        (array) Cutted spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.

        (array) New x-axis( if sub_mode = 'removed') or the intial one(if sub_mode = 'zero').
                Array shape = (n_pixels, ).
    """
    sp = sp.copy()
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    # conversion to indexes
    i_start = (np.abs(wn - wn_start)).argmin()
    i_end = (np.abs(wn - wn_end)).argmin()

    intensity_left = sp[:, i_start]
    intensity_right = sp[:, i_end]

    # intensity gap determination
    intensity_gap = intensity_left - intensity_right
    intensity_gap = np.expand_dims(intensity_gap, axis=1)

    if sub_mode not in ['zero', 'remove']:
        raise ValueError('Invalid sub_mode, must be \'zero\' or \'remove\'')

    if keep_it:
        if sub_mode == 'zero':
            sp[:, :i_start] = 0  # sets the left side part of the spectral cut to zero
            sp[:, i_end + 1:] = 0  # sets the right side part of the spectral cut to zero

        elif sub_mode == 'remove':
            sp = sp[:, i_start:i_end + 1]
            wn = wn[i_start:i_end + 1]
        return sp, wn
    else:
        if sub_mode == 'zero':
            # preserved part on the left side of the spectral cut
            sp[:, i_start:i_end+1] = 0
            return sp, wn

        elif sub_mode == 'remove':
            # preserves the left side part of the spectral cut
            sp_l = sp[:, 0:i_start]
            wn_l = wn[0:i_start]
            # preserves the right side part of the spectral cut
            sp_r = sp[:, i_end+1:]
            wn_r = wn[i_end+1:]
            if reduce_gap:
                # Reduces the gap to zero between the left and right parts
                sp_r = sp_r + intensity_gap

            # remaining parts are joined together
            sp_cut = np.concatenate([sp_l, sp_r], axis=1)
            wn_cut = np.concatenate([wn_l, wn_r])
            return sp_cut, wn_cut


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


def _deprecated_cosmic_filter(sp, ks=3):
    """
    Apply a median filter to the spectrum(s) to remove cosmic rays.

    Note:
        - Former and deprecated version of the cosmic_filter function. It is likely to be removed in
          subsequent BoxSERS versions.

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


if __name__ == "__main__":
    help(__name__)
