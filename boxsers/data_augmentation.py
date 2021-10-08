"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides funtions to generate new spectra by adding different variations to
existing spectra.
"""
import numpy as np
from sklearn.utils import shuffle


def aug_mixup(sp, lab, n_spec=2, alpha=0.5, quantity=1, mode='default', shuffle_enabled=True):
    """
    Randomly generates new spectra by mixing together several spectra with
    a Dirichlet probability distribution.

    This function is inspired of the Mixeup method proposed by zang (Zhang, Hongyi, et al. 2017).

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

        mode(str):{'default', 'review'}, default='default'
            Function mode used.
                -'default': Randomly generates new spectra. Used for data augmentation.
                -'review': Generates spectra with the selected limit values. Used for
                    parameters selection and validation.

        shuffle_enabled : boolean, default=True
            If True, shuffles the new spectra.

    Return:
        (array) New spectra generated.

        (array) New labels generated.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    # lab initialization, lab is forced to be a two-dimensional array
    lab = np.array(lab, ndmin=2)

    sp_len = sp.shape[1]  # spectrum length
    lab_len = lab.shape[1]  # label length
    # defining empty arrays to contain the newly generated data
    sp_aug = np.empty(shape=(1, sp_len))
    lab_aug = np.empty(shape=(1, lab_len))

    # initialization and space allocation
    alpha_array = np.ones(n_spec) * alpha
    # Lambda values generated with a dirichlet distribution
    lam_s = np.random.dirichlet(alpha_array, sp.shape[0])
    arr_review = []
    lab_review = []
    lam_review = []

    if mode == 'review':
        # generation of new spectra
        sp_sum = np.zeros(sp.shape)
        lab_sum = np.zeros(lab.shape)

        for n in range(0, n_spec):
            arr_, lab_ = shuffle(sp, lab)
            lam = np.array(lam_s[:, n], ndmin=2).transpose()
            sp_sum = sp_sum + lam * arr_
            lab_sum = lab_sum + lam * lab_
            arr_review.append(arr_)
            lab_review.append(lab_)
            lam_review.append(lam)
        sp_aug = sp_sum
        lab_aug = lab_sum

        arr_review.append(sp_aug)
        lab_review.append(lab_aug)
        return arr_review, lab_review, lam_review

    elif mode == 'default':

        for i in range(quantity):
            sp_sum = np.zeros(sp.shape)
            lab_sum = np.zeros(lab.shape)

            for n in range(0, n_spec):
                arr_, lab_ = shuffle(sp, lab)
                lam = np.array(lam_s[:, n], ndmin=2).transpose()
                sp_sum = sp_sum + lam * arr_
                lab_sum = lab_sum + lam * lab_

            sp_aug = np.vstack((sp_aug, sp_sum))
            lab_aug = np.vstack((lab_aug, lab_sum))

        # removal of the first empty sample
        sp_aug = np.delete(sp_aug, 0, 0)
        lab_aug = np.delete(lab_aug, 0, 0)

        if shuffle_enabled:
            # spectra and labels are randomly mixed
            sp_aug, lab_aug = shuffle(sp_aug, lab_aug)

    return sp_aug, lab_aug


def aug_multiplier(sp, lab, mult_range, quantity=1, shuffle_enabled=True):
    """
    Randomly generates new spectra with multiplicative factors applied

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        mult_range : integer, list or tuple
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
    # lab initialization, lab is forced to be a two-dimensional array
    lab = np.array(lab, ndmin=2)

    sp_len = sp.shape[1]  # spectrum length
    lab_len = lab.shape[1]  # label length
    # defining empty arrays to contain the newly generated data
    sp_aug = np.empty(shape=(1, sp_len))
    lab_aug = np.empty(shape=(1, lab_len))
    mult_range_inf = 0
    mult_range_sup = 0

    if isinstance(mult_range, list):
        mult_range_inf = mult_range[0]
        mult_range_sup = mult_range[1]
    elif isinstance(mult_range, (float, int)):
        mult_range_inf = 1 - mult_range
        mult_range_sup = 1 + mult_range

    for i in range(quantity):
        # random generation of multiplier(s) using uniform distribution
        multiplier = np.random.uniform(mult_range_inf, mult_range_sup, (sp.shape[0], 1))
        # generation of new spectra
        sp_mult = sp*multiplier
        # new spectra & labels are appended together
        sp_aug = np.vstack((sp_aug, sp_mult))
        lab_aug = np.vstack((lab_aug, lab))

    # first empty sample(first row) is removed from spectra and label
    sp_aug = np.delete(sp_aug, 0, 0)
    lab_aug = np.delete(lab_aug, 0, 0)

    if shuffle_enabled:
        # spectra and labels are randomly mixed
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)

    return sp_aug, lab_aug


def aug_noise(sp, lab, snr=10, quantity=1, noise_type='proportional', shuffle_enabled=True, return_noise=False):
    """
    Randomly generates new spectra with Gaussian noise added.

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

        (array) Last noise signal.
    """
    # sp initialization, sp is forced to be a two-dimensional array
    sp = np.array(sp, ndmin=2)
    # lab initialization, lab is forced to be a two-dimensional array
    lab = np.array(lab, ndmin=2)

    sp_len = sp.shape[1]  # spectrum length
    lab_len = lab.shape[1]  # label length
    # defining empty arrays to contain the newly generated data
    sp_aug = np.empty(shape=(1, sp_len))
    lab_aug = np.empty(shape=(1, lab_len))

    # proportional noise varies with the intensity values of the spectra
    sp_abs = abs(sp)

    # uniform noise varies with the average intensity of the spectra
    sp_avg = np.mean(sp, axis=1, keepdims=True)

    # print(f"{sp_abs=}")  # Debug lines
    # print(f"{sp_avg=}")

    # Converts to dB
    sp_avg_db = 10 * np.log10(sp_avg)
    sp_abs_db = 10 * np.log10(sp_abs)

    # Calculate noise in dB
    noise_avg_db = sp_avg_db - snr
    noise_abs_db = sp_abs_db - snr

    # Convert noise to spectra intensity values
    noise_avg = 10 ** (noise_avg_db / 10)
    noise_abs = 10 ** (noise_abs_db / 10)

    if noise_type == 'proportional':
        for i in range(quantity):
            # generation of new spectra
            noise = np.random.normal(0, noise_abs)
            # new spectra & labels are appended together
            sp_aug = np.vstack((sp_aug, sp+noise))
            lab_aug = np.vstack((lab_aug, lab))
    elif noise_type == 'uniform':
        for i in range(quantity):
            # generation of new spectra
            noise = np.random.normal(0, noise_avg, size=sp.shape)
            # new spectra & labels are appended together
            sp_aug = np.vstack((sp_aug, sp+noise))
            lab_aug = np.vstack((lab_aug, lab))

    # removal of the first empty sample
    sp_aug = np.delete(sp_aug, 0, 0)
    lab_aug = np.delete(lab_aug, 0, 0)

    if shuffle_enabled:
        # spectra and labels are randomly mixed
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)

    if return_noise:
        # added noise is returned
        return sp_aug, lab_aug, noise
    else:
        return sp_aug, lab_aug


def aug_offset(sp, lab, offset_range, quantity=1, shuffle_enabled=True):
    """
    Randomly generates new spectra shifted in intensity.

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        offset_range : integer, list or tuple
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
    # lab initialization, lab is forced to be a two-dimensional array
    lab = np.array(lab, ndmin=2)

    sp_len = sp.shape[1]  # spectrum length
    lab_len = lab.shape[1]  # label length
    # defining empty arrays to contain the newly generated data
    sp_aug = np.empty(shape=(1, sp_len))
    lab_aug = np.empty(shape=(1, lab_len))
    # average intensity is calculated for each spectra
    sp_mean = np.mean(sp, axis=1, keepdims=True)

    offset_range_inf = 0
    offset_range_sup = 0

    if isinstance(offset_range, list):
        offset_range_inf = offset_range[0]
        offset_range_sup = offset_range[1]
    elif isinstance(offset_range, (float, int)):
        offset_range_inf = -offset_range
        offset_range_sup = offset_range

    for i in range(quantity):
        # random generation of intensity offset(s) using uniform distribution
        offset = np.random.uniform(offset_range_inf, offset_range_sup, (sp.shape[0], 1))
        sp_offset = sp + offset*sp_mean
        # new spectra & labels are appended together
        sp_aug = np.vstack((sp_aug, sp_offset))
        lab_aug = np.vstack((lab_aug, lab))

    # first empty sample(first row) is removed from spectra and label
    sp_aug = np.delete(sp_aug, 0, 0)
    lab_aug = np.delete(lab_aug, 0, 0)

    if shuffle_enabled:
        # spectra and labels are randomly mixed(set False for review)
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)
    return sp_aug, lab_aug


def aug_xshift(sp, lab, xshift_range, quantity=1, fill_mode='edge', fill_value=0, shuffle_enabled=True):
    """
    Randomly generates new spectra shifted in wavelength.

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        xshift_range : integer, list or tuple
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
    # lab initialization, lab is forced to be a two-dimensional array
    lab = np.array(lab, ndmin=2)

    # defining a first empty sample for the stacking of the newly generated data
    sp_aug = np.empty(shape=(1, sp.shape[1]))
    lab_aug = np.empty(shape=(1, lab.shape[1]))
    xshift_range_inf = 0
    xshift_range_sup = 0

    if isinstance(xshift_range, (list, tuple)):  # xshift_range is a list or a tuple
        xshift_range_inf = xshift_range[0]
        xshift_range_sup = xshift_range[1]
    elif isinstance(xshift_range, (float, int)):  # xshift_range is a float or a int value
        xshift_range_inf = -xshift_range
        xshift_range_sup = xshift_range

    xshift_range = list(range(xshift_range_inf, xshift_range_sup + 1))

    # the null shift value is removed
    if 0 in xshift_range:
        xshift_range.remove(0)

    for i in range(quantity):
        # random shift(s) generation using an uniform random distribution
        wshft = np.random.choice(xshift_range, (sp.shape[0], 1))
        sp_wshft = _xshift(sp, wshft, fill_mode=fill_mode, fill_value=fill_value)
        # new spectra & labels are appended together
        sp_aug = np.vstack((sp_aug, sp_wshft))
        lab_aug = np.vstack((lab_aug, lab))

    # removal of the first empty sample
    sp_aug = np.delete(sp_aug, 0, 0)
    lab_aug = np.delete(lab_aug, 0, 0)

    if shuffle_enabled:
        # spectra and labels are randomly mixed
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)

    return sp_aug, lab_aug


def _xshift(sp, x_shft, fill_mode, fill_value):
    """
    Moves spectrum/spectra along the pixel axis. Uses the same parameters as the aug_xshift function.
    """

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


def aug_linslope(sp, lab, slope_range, xinter_range, yinter_range=0, quantity=1, shuffle_enabled=True):
    """
    Randomly generates new spectra with additional linear slopes.

    Parameters:
        sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        slope_range : integer, list or tuple
            Values delimiting the range of possible values for random slope. These values can be
            specified as follows:
                - slope_range = (a, b) or [a, b] --> a is the left limit value and b is the right limit value.
                - slope_range = a --> -a is the left limit value and +a is the right limit value.

        xinter_range : integer, list or tuple
            Values delimiting the possible random values for the x-intersept. These values are specified the
            same way as "slope_range".

        yinter_range : integer, list or tuple
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
    # lab initialization, lab is forced to be a two-dimensional array
    lab = np.array(lab, ndmin=2)

    sp_len = sp.shape[1]   # spectrum length
    lab_len = lab.shape[1]  # label length
    # defining empty arrays to contain the newly generated data
    sp_aug = np.empty(shape=(1, sp_len))
    lab_aug = np.empty(shape=(1, lab_len))
    pixels = np.arange(0, sp_len)
    # average intensity is calculated for each spectra
    sp_mean = np.mean(sp, axis=1, keepdims=True)

    if isinstance(slope_range, list):
        slope_range_inf = slope_range[0]
        slope_range_sup = slope_range[1]
    elif isinstance(slope_range, (float, int)):
        slope_range_inf = -slope_range
        slope_range_sup = slope_range

    if isinstance(xinter_range, list):
        xinter_range_inf = xinter_range[0]
        xinter_range_sup = xinter_range[1]
    elif isinstance(xinter_range, (float, int)):
        xinter_range_inf = -xinter_range
        xinter_range_sup = xinter_range

    if isinstance(yinter_range, list):
        yinter_range_inf = yinter_range[0]
        yinter_range_sup = yinter_range[1]
    elif isinstance(yinter_range, (float, int)):
        yinter_range_inf = -yinter_range
        yinter_range_sup = +yinter_range

    for i in range(quantity):
        # random slope and intercept(s) generation using uniform distribution
        slope = np.random.uniform(slope_range_inf, slope_range_sup, (sp.shape[0], 1)) * sp_mean
        xinter = -np.random.uniform(xinter_range_inf, xinter_range_sup, (sp.shape[0], 1)) * sp_len
        yinter = np.random.uniform(yinter_range_inf, yinter_range_sup, (sp.shape[0], 1)) * sp_mean
        # generation of new spectra
        sp_slope = sp + ((pixels+xinter)*slope/sp_len + yinter)
        # new spectra & labels are appended together
        sp_aug = np.vstack((sp_aug, sp_slope))
        lab_aug = np.vstack((lab_aug, lab))

    # removal of the first empty sample
    sp_aug = np.delete(sp_aug, 0, 0)
    lab_aug = np.delete(lab_aug, 0, 0)

    if shuffle_enabled:
        # spectra and labels are randomly mixed
        sp_aug, lab_aug = shuffle(sp_aug, lab_aug)

    return sp_aug, lab_aug


if __name__ == "__main__":
    help(__name__)
