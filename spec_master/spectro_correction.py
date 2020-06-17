import numpy as np
from scipy import interpolate, sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter


def spectral_normalization(sp):
    """
    Normalizes the spectrum(s) individually so that the max value is 1 and the min value is 0.

    Parameters:
        sp: Input spectrum or spectra array(1 spectrum/row).

    Returns:
        ndarray:  Normalized spectrum or spectra
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    # max/min calculation for each spectrum
    sp_max = np.max(sp, axis=1, keepdims=True)
    sp_min = np.min(sp, axis=1, keepdims=True)
    # normalization operation
    sp_norm = (sp-sp_min)/(sp_max-sp_min)
    return sp_norm


def spline_interpolation(sp, wn, new_wn, multiple_wn=False):
    """
    Performs a one-dimensional interpolation spline on a given spectrum to reproduce it with a new wavelength
    or wavelength number.

    Parameters:
        sp: Input spectrum or spectrum array(spectra/row).
        wn: Current wavenumber or wavelenght values.
        new_wn: New wavenumber or wavelenght values chosen.
        multiple_wn(bool): Bool if there is different wavenumber values

    Returns:
        ndarray: new_sp
    """
    new_sp = np.zeros((sp.shape[0], len(new_wn)))
    error_ind = []  # initialize a list to stock indexes where the errors occurred
    value_error_detected_event = 0

    if multiple_wn:
        if wn.shape[0] != sp.shape[0]:
            print('sizes do not fit')
        else:
            for i in range(sp.shape[0]):
                try:
                    s = interpolate.InterpolatedUnivariateSpline(wn[i], sp[i], ext=1)
                    new_sp[i] = s(new_wn)
                except ValueError:
                    # Display a message at the end of the function if an error has occurred.
                    value_error_detected_event = 1
                    # Saves the indexes where the errors occurred
                    error_ind.append(i)
    else:
        for i in range(sp.shape[0]):
            s = interpolate.InterpolatedUnivariateSpline(wn, sp[i], ext=1)
            new_sp[i] = s(new_wn)

    if value_error_detected_event == 1:
        new_sp = np.delete(new_sp, error_ind, 0)
        print('Some spectra raised errors and were removed')
        print('Erroneous spectrum indexes are identified by the second output of this function.')
        print('Delete the corresponding label with this command: new_label = np.delete(label, erroneous_index)')

    return new_sp, error_ind


def baseline_subtraction(sp, lam=10000, p=0.001, niter=10):
    """ Applies an ALS baseline correction to the spectra.
        Update April 2019: improved computing speed

    Parameters:
        sp: Input spectrum or spectrum array(spectra/row).
        lam: 2nd derivative constraint.
        p: Weighting of positive residuals.
        niter: Maximum number of iterations.

    Returns:
        ndarray:  (baseline array, baseline substracted spectrum or spectra array)
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)

    # initialization and space allocation
    z = np.zeros(sp.shape)  # baseline signal array
    sp_length = sp.shape[1]  # length of a spectrum
    diag = sparse.diags([1, -2, 1], [0, -1, -2], shape=(sp_length, sp_length - 2))
    diag = lam * diag.dot(diag.transpose())
    w = np.ones(sp_length)
    w_mat = sparse.spdiags(w, 0, sp_length, sp_length)

    for n in range(0, len(sp)):
        for i in range(niter):
            w_mat.setdiag(w)
            zz = w_mat + diag
            z[n] = spsolve(zz, w * sp[n])
            w = p * (sp[n] > z[n]) + (1 - p) * (sp[n] < z[n])  # w is updated according to z
    return sp - z, z


def spectral_cut(sp, wn, wn_start, wn_end):
    """
    Subtracts a delimited part of the spectrum or spectra and joins the remaining parts together

    Parameters:
        sp: Input spectrum or spectrum array(spectra/row).
        wn: Wavenumber or wavelenght.
        wn_start: Start(with the same unit as wn) of the subtracted spectral region.
        wn_end: End(with the same unit as wn) of the subtracted spectral region.

    Returns:
        ndarray:  sp_cut, wn_cut
    """
    # sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    # conversion to indexes
    i_start = (np.abs(wn - wn_start)).argmin()
    i_end = (np.abs(wn - wn_end)).argmin()
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


def savgol_smoothing(sp, window_length, p=3, degree=0):
    """
    Smoothes the spectra with a Savitzky-Golay filter.

    Parameters:
        sp: Spectra array.
        window_length (odd int)(min_value=5): Savitzky-Golay filters window length.
        p(int): Savitzky-Golay polynomial order.
        degree(int): Savitzky-Golay derivative order.

    Returns: sp_svg
    """
    sp_svg = savgol_filter(sp, window_length, polyorder=p, deriv=degree)

    return sp_svg
