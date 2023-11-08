"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides funtions to generate new spectra by adding different variations to
existing spectra.
"""
import numpy as np
from sklearn.utils import shuffle


def _range_converter(param_range, multiplier_cond=False):
    """
    Returns superior and inferior bounds according to a given range.

    Parameters:
        param_range : float or integer, list or tuple
            Range of possible values for a given value. These values can be specified as follows:
                - param_range = (a, b) or [a, b] --> a is the left limit value and b is the right limit value.
                - (multiplier_cond=False)
                    - param_range = a --> -a is the left limit value and +a is the right limit value.
                - (multiplier_cond=True)
                    - param_range = a --> (1 - a) is the left limit value and (1 + a) is the right limit value.

        multiplier_cond : boolean, default=False
            See above for information

    Returns:
        (float, int) Inferior boudary value.

        (float, int) Superior boundary value.
    """
    param_range_inf = 0.0
    param_range_sup = 0.0

    if isinstance(param_range, (list, tuple)):
        param_range_inf = param_range[0]
        param_range_sup = param_range[1]
    elif isinstance(param_range, (float, int)) and not multiplier_cond:
        param_range_inf = -param_range
        param_range_sup = param_range
    elif isinstance(param_range, (float, int)) and multiplier_cond:
        param_range_inf = 1 - param_range
        param_range_sup = 1 + param_range
    return param_range_inf, param_range_sup


def _xshift(sp, x_shft, fill_mode, fill_value):
    """
    Moves spectrum/spectra along the pixel axis. Uses the same parameters as the aug_xshift function.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)

    # initialization and space allocation for shft_sp and fill_value
    shft_sp = np.zeros_like(sp)
    fill_value_left = np.zeros_like(x_shft)
    fill_value_right = np.zeros_like(x_shft)
    # x_shft index splitting for positive and negative shifts
    positives = np.argwhere(x_shft > 0)[:, 0]
    negatives = np.argwhere(x_shft < 0)[:, 0]

    if fill_mode == 'edge':
        fill_value_left = sp[:, 0]
        fill_value_right = sp[:, -1]
    elif fill_mode == 'fixed':
        fill_value_left[:] = fill_value
        fill_value_right[:] = fill_value
    else:
        print('Invalid fill_mode: fill value(s) has been set to zero')

    for p in positives:
        shft_sp[p, :int(x_shft[p])] = fill_value_left[p]
        shft_sp[p, int(x_shft[p]):] = sp[p, :-int(x_shft[p])]
    for n in negatives:
        shft_sp[n, int(x_shft[n]):] = fill_value_right[n]
        shft_sp[n, :int(x_shft[n])] = sp[n, -int(x_shft[n]):]
    return shft_sp


def aug_mixup(sp, lab, n_spec=2, alpha=0.5, quantity=1, shuffle_enabled=True, return_infos=False):
    """
    Randomly generates new spectra by mixing together several spectra with
    a Dirichlet probability distribution.

    This function is inspired of the Mixeup method proposed by zang (Zhang, Hongyi, et al. 2017).

    Notes:
        Updated [2023-05-31]:
            - parameter `mode` removed, use `return_infos` instead for parameters selection and validation.
            - Computation time and memory consumption reduced !

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels(must be binary) assigned the "sp" spectra, array shape = (n_spectra, n_classes).

        n_spec: integer, default=2
            Amount of spectrum mixed together.

        alpha : float
            Dirichlet distribution concentration parameter.

        quantity : integer, default=1
            Quantity of new spectra generated for one spectrum. If less than or equal to zero, no new
            spectrum is generated.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

        return_infos : boolean, default=False
            If True, returns the indexes and the lambda values of the spectra mixed together

    Return:
        (array) New spectra generated.

        (array) New labels generated.

        (array) Optional; Indexes of the spectra mixed together.

        (array) Optional; Lambda values of the spectra mixed together.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    if lab.ndim == 1:
        # lab is forced to be a two-dimensional array
        lab = np.expand_dims(lab, axis=1)

    n_spectra, sp_len = sp.shape  # number of spectra, spectrum length
    lab_len = lab.shape[1]  # label length

    # array preallocation
    sp_aug = np.zeros((quantity * n_spectra, sp_len))
    lab_aug = np.zeros((quantity * n_spectra, lab_len))

    # initialization and space allocation
    alpha_array = np.ones(n_spec) * alpha
    # Lambda values generated with a dirichlet distribution
    lambda_values = np.random.dirichlet(alpha_array, quantity*n_spectra)

    # random spectra index selection
    random_indexes = np.random.choice(n_spectra, size=(quantity * n_spectra, n_spec), replace=True)

    for i, (lam, index) in enumerate(zip(lambda_values, random_indexes)):
        mixed_sp = lam[:, np.newaxis] * sp[index]
        mixed_lab = lam[:, np.newaxis] * lab[index]
        sp_aug[i] += np.sum(mixed_sp, axis=0)
        lab_aug[i] += np.sum(mixed_lab, axis=0)

    if shuffle_enabled:
        # spectra, labels, indexes and lambda are randomly shuffled
        sp_aug, lab_aug, random_indexes, lambda_values = shuffle(sp_aug, lab_aug, random_indexes, lambda_values)

    if return_infos:
        # returns the indexes for the spectra mixed together
        return sp_aug, lab_aug, random_indexes, lambda_values

    return sp_aug, lab_aug


def aug_newband(sp, lab, inv_p=None, inv_p_degree=1, intensity_range=(0.05, 0.95), width_range=(2, 5),
                quantity=1, shuffle_enabled=True, return_band=False):
    """
    Randomly generates new spectra with additionnal Gaussian peak added.

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        inv_p: string, default=None,
            Reference value for the inverse probability density function. If None, it is the input spectra
            that will be used.

        inv_p_degree : float, default=1
            Determines where additional Raman bands will be more likely positioned on the spectrum according
            to inv_p.
                - > 0 : positioned at different locations than input spectrum bands.
                - = 0 : randomly postioned.
                - < 0 : positioned close to input spectrum bands.

        intensity_range : float or integer, list or tuple, default = (0.05, 0.95)
            Values delimiting the range of possible values for the Raman band intensity..

        width_range : list or tuple, default = (2, 5)
            Values delimiting the range of possible values for the Raman band width (in pixels).

        quantity : integer, default=1
            Quantity of new spectra generated for one spectrum. If less than or equal to zero, no new
            spectrum is generated.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

        return_band : boolean, default=False
            If True, returns the individual peak added.

    Return:
        (array) New spectra generated.

        (array) New labels generated.

        (array) Optional; New Raman bands added.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    if lab.ndim == 1:
        # lab is forced to be a two-dimensional array
        lab = np.expand_dims(lab, axis=1)

    n_spectra, sp_len = sp.shape  # number of spectra, spectrum length

    # array preallocation
    sp_aug = np.empty((quantity * n_spectra, sp_len))
    added_bands = np.empty((quantity * n_spectra, sp_len))
    # as no changes are made to the labels, they are repeated to match the new spectra generated.
    lab_aug = np.repeat(lab, quantity, axis=0)

    if inv_p is None:
        # spectra values are used as the input for the inverse probability density
        inv_p = np.abs(sp)  # np.abs() used to remove negatives values
    else:
        # inverse probability density function is given
        inv_p = np.array(inv_p, ndmin=2)

    # inverse density probabilities calculation
    inv_density = 1.0 / (inv_p ** inv_p_degree)
    # normalizing the probabilities to sum up to 1 for each row
    inverse_density_norm = inv_density / inv_density.sum(axis=1, keepdims=1)
    inverse_density_norm = np.repeat(inverse_density_norm, quantity, axis=0)

    # upper and lower bounds are determined from the given range
    intensity_range_inf, intensity_range_sup = _range_converter(intensity_range)

    # upper and lower bounds are multiplied by the average intensity of each spectrum
    intensity_inf = np.mean(sp, axis=1) * intensity_range_inf
    intensity_sup = np.mean(sp, axis=1) * intensity_range_sup

    # random band_intensities and band_widths generation using uniform distribution
    band_intensities = np.random.uniform(intensity_inf.repeat(quantity), intensity_sup.repeat(quantity))
    band_widths = np.random.uniform(width_range[0]/2, width_range[1]/2, size=(quantity * n_spectra))

    indexes = np.arange(sp_len)
    for i, (intensity, width) in enumerate(zip(band_intensities, band_widths)):
        # choose a random index based on the inverse density probabilities
        random_index = np.random.choice(sp_len, p=inverse_density_norm[i])
        # generates a new Raman Gaussian band
        new_band = intensity * np.exp(-0.5 * ((indexes - random_index) / width) ** 2)
        # generation of new spectra
        sp_newband = sp[i // quantity] + new_band

        # sp_aug and added band are filled progressively
        sp_aug[i] = sp_newband
        added_bands[i] = new_band

    if shuffle_enabled:
        # spectra, labels and new bands are randomly shuffled
        sp_aug, lab_aug, added_bands = shuffle(sp_aug, lab_aug, added_bands)

    if return_band:
        # the new Raman peak is also returned
        return sp_aug, lab_aug, added_bands
    return sp_aug, lab_aug


def aug_noise(sp, lab, snr=10, quantity=1, noise_type='proportional', shuffle_enabled=True, return_noise=False):
    """
    Randomly generates new spectra with Gaussian noise added.

    Notes:
        Updated [2023-05-31]:
            - Computation time and memory consumption reduced !

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        snr : positive float or int
            Signal-to-noise ratio (SNR)

        quantity : integer, default=1
            Quantity of new spectra generated for one spectrum. If less than or equal to zero, no new
            spectrum is generated.

        noise_type : {'proportional', 'uniform'}, default='proportional'
            Type of noise added to the spectra.
                - 'proportional': The standard deviation of the noise varies over a spectrum and is proportional
                    to the intensity of each pixel.
                - 'uniform': The standard deviation of the noise is uniform over a spectrum and is proportional
                    to the average intensity of each spectrum.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

        return_noise : boolean, default=False
            If True, returns the last generated noise signal

    Return:
        (array) New spectra generated.

        (array) New labels generated.

        (array) Optional; Noise generated.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    if lab.ndim == 1:
        # lab is forced to be a two-dimensional array
        lab = np.expand_dims(lab, axis=1)

    n_spectra, sp_len = sp.shape  # number of spectra, spectrum length

    # array preallocation
    sp_aug = np.empty((quantity * n_spectra, sp_len))
    # as no changes are made to the labels, they are repeated to match the new spectra generated.
    lab_aug = np.repeat(lab, quantity, axis=0)

    if noise_type == 'proportional':
        # proportional noise varies with the intensity values of the spectra
        std_ref = abs(sp)  # shape = (n_spectra, sp_len)
    elif noise_type == 'uniform':
        # uniform noise varies with the average intensity of the spectra
        std_ref = np.mean(sp, axis=1, keepdims=True)  # shape = (n_spectra, 1)
    else:
        raise ValueError('Invalid noise_type, valid choices: {\'proportional\', \'uniform\'}')

    # Converts to dB
    std_db = 10 * np.log10(std_ref)

    # Calculate noise in dB
    noise_db = std_db - snr

    # Convert noise to spectra intensity values
    noise_std = 10 ** (noise_db / 10)
    noise_std = np.repeat(noise_std, quantity, axis=0)

    noises = np.random.normal(0, noise_std, size=sp_aug.shape)

    for i, noise in enumerate(noises):
        # generation of new spectra
        sp_noise = sp[i // quantity] + noise
        # sp_aug is filled progressively
        sp_aug[i] = sp_noise

    if shuffle_enabled:
        # spectra, labels and noise are randomly mixed
        sp_aug, lab_aug, noises = shuffle(sp_aug, lab_aug, noises)

    if return_noise:
        # added noise is returned
        return sp_aug, lab_aug, noises
    return sp_aug, lab_aug


def aug_multiplier(sp, lab, mult_range, quantity=1, shuffle_enabled=True):
    """
    Randomly generates new spectra with multiplicative factors applied

    Notes:
        Updated [2023-05-31]:
            - Computation time and memory consumption reduced !

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        mult_range : float or integer, list or tuple
            Values delimiting the range of possible values for random multiplier. These values can be
            specified as follows:
                - mult_range = (a, b) or [a, b] --> a is the left limit value and b is the right limit value.
                - mult_range = a --> (1 - a) is the left limit value and (1 + a) is the right limit value.

        quantity : integer, default=1
            Quantity of new spectra generated for one spectrum. If less than or equal to zero, no new
            spectrum is generated.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

    Return:
        (array) New spectra generated.

        (array) New labels generated.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    if lab.ndim == 1:
        # lab is forced to be a two-dimensional array
        lab = np.expand_dims(lab, axis=1)

    n_spectra, sp_len = sp.shape  # number of spectra, spectrum length

    # array preallocation
    sp_aug = np.empty((quantity * n_spectra, sp_len))
    # as no changes are made to the labels, they are repeated to match the new spectra generated.
    lab_aug = np.repeat(lab, quantity, axis=0)

    # upper and lower bounds are determined from the given range
    mult_range_inf, mult_range_sup = _range_converter(mult_range, multiplier_cond=True)

    # random generation of multiplier(s) using uniform distribution
    multipliers = np.random.uniform(mult_range_inf, mult_range_sup, size=(n_spectra * quantity))

    for i, multiplier in enumerate(multipliers):
        # generation of new spectra
        sp_mult = sp[i // quantity] * multiplier
        # sp_aug is filled progressively
        sp_aug[i] = sp_mult

    if shuffle_enabled:
        # spectra and labels are randomly mixed
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)

    return sp_aug, lab_aug


def aug_offset(sp, lab, offset_range, quantity=1, shuffle_enabled=True):
    """
    Randomly generates new spectra shifted in intensity.

    Notes:
        Updated [2023-05-31]:
            - Computation time and memory consumption reduced !

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        offset_range : float or integer, list or tuple
            Values delimiting the range of possible values for random intensity offset. These values can be
            specified as follows:
                - offset_range = (a, b) or [a, b] --> a is the left limit value and b is the right limit value.
                - offset_range = a --> -a is the left limit value and +a is the right limit value.

        quantity : integer, default=1
            Quantity of new spectra generated for one spectrum. If less than or equal to zero, no new
            spectrum is generated.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

    Return:
        (array) New spectra generated.

        (array) New labels generated.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    if lab.ndim == 1:
        # lab is forced to be a two-dimensional array
        lab = np.expand_dims(lab, axis=1)

    n_spectra, sp_len = sp.shape  # number of spectra, spectrum length

    # array preallocation
    sp_aug = np.empty((quantity * n_spectra, sp_len))
    # as no changes are made to the labels, they are repeated to match the new spectra generated.
    lab_aug = np.repeat(lab, quantity, axis=0)

    # upper and lower bounds are determined from the given range
    offset_range_inf, offset_range_sup = _range_converter(offset_range)

    # upper and lower bounds are multiplied by the average intensity of each spectrum
    offset_range_inf = offset_range_inf * np.mean(sp, axis=1)
    offset_range_sup = offset_range_sup * np.mean(sp, axis=1)

    # random intensity offset(s) generation using uniform distribution
    offsets = np.random.uniform(offset_range_inf.repeat(quantity), offset_range_sup.repeat(quantity))

    for i, offset in enumerate(offsets):
        # generation of new spectra
        sp_offset = sp[i // quantity] + offset
        # sp_aug is filled progressively
        sp_aug[i] = sp_offset

    if shuffle_enabled:
        # spectra and labels are randomly mixed(set False for review)
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)
    return sp_aug, lab_aug


def aug_linslope(sp, lab, slope_range, xinter_range, yinter_range=0, quantity=1, shuffle_enabled=True):
    """
    Randomly generates new spectra with additional linear slopes.

    Notes:
        Updated [2023-05-31]:
            - Computation time and memory consumption reduced !

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        slope_range : float or integer, list or tuple
            Values delimiting the range of possible values for random slope. These values can be
            specified as follows:
                - slope_range = (a, b) or [a, b] --> a is the left limit value and b is the right limit value.
                - slope_range = a --> -a is the left limit value and +a is the right limit value.

        xinter_range : float or integer, list or tuple
            Values delimiting the possible random values for the x-intersept. These values are specified the
            same way as "slope_range". If = 0, the x-intercept will be at the left end of the spectrum, and
            if =1 at the right end.

        yinter_range : float or integer, list or tuple
            Values delimiting the possible random values for the y-intersept. Same effect as adding an offset
            with the function "aug_ioffset". These values are specified the same way as "slope_range".

        quantity : integer, default=1
            Quantity of new spectra generated for one spectrum. If less than or equal to zero, no new
            spectrum is generated.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

    Return:
        (array) New spectra generated.

        (array) New labels generated.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    if lab.ndim == 1:
        # lab is forced to be a two-dimensional array
        lab = np.expand_dims(lab, axis=1)

    n_spectra, sp_len = sp.shape  # number of spectra, spectrum length

    # array preallocation
    sp_aug = np.empty((quantity * n_spectra, sp_len))
    # as no changes are made to the labels, they are repeated to match the new spectra generated.
    lab_aug = np.repeat(lab, quantity, axis=0)

    # average intensity is calculated for each spectra
    sp_mean = np.mean(sp, axis=1)

    # upper and lower bounds are determined from the given range
    slope_range_inf, slope_range_sup = _range_converter(slope_range)
    xinter_range_inf, xinter_range_sup = _range_converter(xinter_range)
    yinter_range_inf, yinter_range_sup = _range_converter(yinter_range)

    # upper and lower bounds are multiplied by the average intensity of each spectrum (slope and y_intercept only)
    slope_range_inf = slope_range_inf * sp_mean
    slope_range_sup = slope_range_sup * sp_mean
    yinter_range_inf = yinter_range_inf * sp_mean
    yinter_range_sup = yinter_range_sup * sp_mean
    # upper and lower bounds are multiplied by the spectrum lenght
    xinter_range_inf = xinter_range_inf * sp_len
    xinter_range_sup = xinter_range_sup * sp_len

    # random slope(s) and intercept(s) generation using uniform distribution
    slopes = np.random.uniform(slope_range_inf.repeat(quantity), slope_range_sup.repeat(quantity))
    xinters = -1 * np.random.uniform(xinter_range_inf, xinter_range_sup, size=(quantity * n_spectra))
    yinters = np.random.uniform(yinter_range_inf.repeat(quantity), yinter_range_sup.repeat(quantity))

    indexes = np.arange(sp_len)
    for i, (slope, xinter, yinter) in enumerate(zip(slopes, xinters, yinters)):
        # generation of new spectra
        sp_slope = sp[i // quantity] + ((indexes + xinter) * slope / sp_len + yinter)
        # sp_aug is filled progressively
        sp_aug[i] = sp_slope

    if shuffle_enabled:
        # spectra and labels are randomly mixed
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)

    return sp_aug, lab_aug


def aug_xshift(sp, lab, xshift_range, quantity=1, fill_mode='edge', fill_value=0, shuffle_enabled=True):
    """
    Randomly generates new spectra shifted in wavelength.

    Notes:
        Updated [2023-05-31]:
            - Computation time and memory consumption reduced !

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        xshift_range : float or integer, list or tuple
            Values delimiting the range of possible values for random pixel shift. These values can be
            specified as follows:
                - xshift_range = (a, b) or [a, b] --> a is the left limit value and b is the right limit value.
                - xshift_range = a --> -a is the left limit value and +a is the right limit value.

        quantity : integer, default=1
            Quantity of new spectra generated for one spectrum. If less than or equal to zero, no new
            spectrum is generated.

        fill_mode : {'edge', 'fixed'}, default='edge'
            Fill mode used for new values created at the extremities of the spectra when they are shifted.
                - 'edge': Edge values are used to fill new values.
                - 'fixed': A fixed value(fixed_value) is used to fill new values.

        fill_value : integer or float, default=0
            Value used when "fill_mode" is 'fixed'.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

    Return:
        (array) New spectra generated.

        (array) New labels generated.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    if lab.ndim == 1:
        # lab is forced to be a two-dimensional array
        lab = np.expand_dims(lab, axis=1)

    n_spectra, sp_len = sp.shape  # number of spectra, spectrum length

    # array preallocation
    sp_aug = np.empty((quantity * n_spectra, sp_len))
    # as no changes are made to the labels, they are repeated to match the new spectra generated.
    lab_aug = np.repeat(lab, quantity, axis=0)

    # upper and lower bounds are determined from the given range
    xshift_range_inf, xshift_range_sup = _range_converter(xshift_range)

    # the null shift value is removed
    xshift_range_cor = list(range(xshift_range_inf, xshift_range_sup + 1))
    if 0 in xshift_range_cor:
        xshift_range_cor.remove(0)

    # random shift(s) generation using an uniform random distribution
    xshifts = np.random.choice(xshift_range_cor, size=(quantity * n_spectra, 1), replace=True)

    for i, xshift in enumerate(xshifts):
        # generation of new spectra
        sp_xshift = _xshift(sp[i//quantity], xshift, fill_mode=fill_mode, fill_value=fill_value)
        # sp_aug is filled progressively
        sp_aug[i] = sp_xshift

    if shuffle_enabled:
        # spectra and labels are randomly mixed
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)

    return sp_aug, lab_aug


if __name__ == "__main__":
    help(__name__)
