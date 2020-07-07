import numpy as np
from sklearn.utils import shuffle


class SpecAugPipeline:

    def __init__(self, sp, lab):
        """
        Parameters:
            sp : Input spectrum or spectrum array(spectra/row).
            lab : Label associated to sp.

        Notes:
            Self.sp_aug and self.lab_aug contain the newly generated and  initial spectra.

            It is from self.sp and self.lab that the new spectra are generated.
        """
        # sp is forced to be a two-dimensional array
        self.sp = np.array(sp, ndmin=2)

        if lab.ndim == 1:  # lab is a tuple
            # lab is converted to a column vector shape = [x,1]
            self.lab = np.reshape(lab, (lab.shape[0], 1))
        else:
            self.lab = lab

        self.sp_aug = self.sp
        self.lab_aug = self.lab

    def out(self):
        return self.sp_aug, self.lab_aug

    def return_base_sizes(self):
        return self.sp.shape, self.lab.shape

    def return_aug_sizes(self):
        return self.sp_aug.shape, self.lab_aug.shape

    def updtate_base(self, new_sp=None, new_lab=None):

        if new_sp and new_lab is None:
            print('ss')

        self.sp = self.sp_aug
        self.lab = self.lab_aug

    @staticmethod
    def noise_addition(sp, noise_std):
        """
        Add gaussian noise

        Parameters:
            sp: Input spectrum or spectrum array(spectra/row).
            noise_std(int or float): Noise standard deviation.

        Returns:
            (ndarray): Spectrum or spectra with noise added
        """
        return np.random.normal(sp, noise_std)

    def aug_noise(self, noise_std, iterations=1, update=False, _debugcheck=False):
        """
        Description

        Parameters:
            noise_std:
            iterations(int):
            noise_std(float or None):
            update(bool):
            _debugcheck:

        Return:
            (ndarray, ndarray): Augmented spectrum and label arrays
        """
        for i in range(iterations):
            # generation of new spectra
            noise_sp = self.noise_addition(self.sp, noise_std)
            # new spectra & labels are appended to the original ones
            self.sp_aug = np.vstack((self.sp_aug, noise_sp))
            self.lab_aug = np.vstack((self.lab_aug, self.lab))
        # spectra and labels are randomly mixed
        self.sp_aug, self.lab_aug = shuffle(self.sp_aug, self.lab_aug)

        if _debugcheck:
            return noise_sp
        if update:
            # overwrites source data with augmented data
            self.sp = self.sp_aug
            self.lab = self.lab_aug

        return self.sp_aug, self.lab_aug

    @staticmethod
    def ioffset(sp, offset):
        """
        Adds a uniform intensity offset to the signal

        Parameters:
            sp: Input spectrum or spectrum array(spectra/row).
            offset(int or float): Offset

        Returns:
            (ndarray): Spectrum or spectra with offset added
        """
        return sp + offset

    def aug_ioffset(self, offset_lim, iterations=1, noise_std=None, update=False, _debugcheck=False):
        """
        Description

        Parameters:
            offset_lim:
            iterations(int):
            noise_std(float or None):
            update(bool):
            _debugcheck:

        Return:
            (ndarray, ndarray): Augmented spectrum and label arrays
        """
        for i in range(iterations):
            # random intensity offset(s) generation using uniform distribution
            arr_offset = np.random.uniform(-offset_lim, offset_lim, (self.sp.shape[0], 1))
            # generation of new spectra
            offset_sp = self.ioffset(self.sp, arr_offset)
            if noise_std is not None:
                # noise is added on the new spectra
                offset_sp = self.noise_addition(offset_sp, noise_std)

            # new spectra & labels are appended to the original ones
            self.sp_aug = np.vstack((self.sp_aug, offset_sp))
            self.lab_aug = np.vstack((self.lab_aug, self.lab))
        # spectra and labels are randomly mixed
        self.sp_aug, self.lab_aug = shuffle(self.sp_aug, self.lab_aug)

        if _debugcheck:
            return offset_sp
        if update:
            # overwrites source data with augmented data
            self.sp = self.sp_aug
            self.lab = self.lab_aug

        return self.sp_aug, self.lab_aug

    @staticmethod
    def wshift(sp, w_shft, fill_mode='edge', value=0):
        """
        Moves spectrum/spectra along the wavelength/Raman shift  axis.

        Parameters:
            sp: Input spectrum or spectrum array(spectra/row).
            w_shft: wavelenth or wavenumber shift.
            fill_mode(str): {'edge', 'fixed'} Filling mode used for new values created when shifting spectra.
                'edge': Edge values are used to fill new values
                'fixed': A fixed value(set_value) is used to fill new values
            value: Filling value used when the filling mode is 'constant'

        Returns:
            ndarray: Spectrum/spectra following the shift(s) on the wavelength axis.

        Notes:
            w_shft must either be a dimensionless array or a column vector and its size must not exceed
            the number of spectra contained in sp.
        """

        # w_shft initialization
        w_shft = np.array(w_shft, ndmin=2)
        # sp_out initialization
        sp = np.array(sp, ndmin=2)
        shft_sp = sp
        # fill value initialization
        fill_value_left = np.zeros_like(w_shft)
        fill_value_right = np.zeros_like(w_shft)
        # w_shft index splitting for positive and negative shifts
        positives = np.argwhere(w_shft > 0)[:, 0]
        negatives = np.argwhere(w_shft < 0)[:, 0]

        if fill_mode == 'edge':
            fill_value_left = sp[:, 0]
            fill_value_right = sp[:, -1]
        elif fill_mode == 'fixed':
            fill_value_left[:] = value
            fill_value_right[:] = value
        else:
            print('Invalid fill_mode: fill value(s) has been set to zero')

        for p in positives:
            shft_sp[p, :int(w_shft[p])] = fill_value_left[p]
            shft_sp[p, int(w_shft[p]):] = sp[p, :-int(w_shft[p])]
        for n in negatives:
            shft_sp[n, int(w_shft[n]):] = fill_value_right[n]
            shft_sp[n, :int(w_shft[n])] = sp[n, -int(w_shft[n]):]
        return shft_sp

    def aug_wshift(self, w_shft_lim, iterations=1, noise_std=None, update=False, _debugcheck=False):
        """
        Description

        Parameters:
            w_shft_lim:
            iterations(int):
            noise_std(float or None):
            update(bool):
            _debugcheck:

        Return:
            (ndarray, ndarray): Augmented spectrum and label arrays
        """

        for i in range(iterations):
            # random shift(s) generation using uniform distribution
            arr_shift = np.random.randint(-w_shft_lim, w_shft_lim, (self.sp.shape[0], 1))
            # generation of new spectra
            shift_sp = self.wshift(self.sp, arr_shift)
            if noise_std is not None:
                # noise is added on the new spectra
                shift_sp = self.noise_addition(shift_sp, noise_std)

            # new spectra & labels are appended to the original ones
            self.sp_aug = np.vstack((self.sp_aug, shift_sp))
            self.lab_aug = np.vstack((self.lab_aug, self.lab))

        # spectra and labels are randomly mixed
        self.sp_aug, self.lab_aug = shuffle(self.sp_aug, self.lab_aug)

        if update:
            # overwrites source data with augmented data
            self.sp = self.sp_aug
            self.lab = self.lab_aug

        if _debugcheck:
            return shift_sp

        return self.sp_aug, self.lab_aug

    @staticmethod
    def linear_slope(sp, slope, inter):
        """
        Adds a linear slope to the signal

        Parameters:
            sp: Input spectrum or spectrum array(spectra/row).
            slope(int or float): Linear slope slope parameter.
            inter(int or float): Linear slope intercept parameter.

        Returns:
            (ndarray): Spectrum or spectra with linear slope added
        """
        sp = np.array(sp, ndmin=2)
        slope_range = np.arange(-(sp.shape[1]) / 2, (sp.shape[1]) / 2)
        return sp + (slope * slope_range + inter)

    def aug_linear_slope(self, slope_lim, inter_lim, iterations=1, noise_std=None, update=False, _debugcheck=False):
        """
        Description

        Parameters:
            slope_lim:
            inter_lim:
            iterations(int):
            noise_std(float):
            update(bool):
            _debugcheck(bool):

        Return:
            (ndarray, ndarray): Augmented spectrum and label arrays
        """
        for i in range(iterations):
            # random slope and intercept(s) generation using uniform distribution
            arr_slope = np.random.uniform(-slope_lim, slope_lim, (self.sp.shape[0], 1))
            arr_inter = np.random.uniform(-inter_lim, inter_lim, (self.sp.shape[0], 1))
            # generation of new spectra
            slope_sp = self.linear_slope(self.sp, arr_slope, arr_inter)
            if noise_std is not None:
                # noise is added on the new spectra
                slope_sp = self.noise_addition(slope_sp, noise_std)

            # new spectra & labels are appended to the original ones
            self.sp_aug = np.vstack((self.sp_aug, slope_sp))
            self.lab_aug = np.vstack((self.lab_aug, self.lab))

        # the spectra and labels are randomly mixed
        self.sp_aug, self.lab_aug = shuffle(self.sp_aug, self.lab_aug)

        if update:
            # overwrites source data with augmented data
            self.sp = self.sp_aug
            self.lab = self.lab_aug

        if _debugcheck:
            return slope_sp

        return self.sp_aug, self.lab_aug

    @staticmethod
    def multiplier(sp, multiplier):
        return sp * multiplier

    def aug_multiplier(self, mult_lim=0.1, iterations=1, noise_std=None, update=False, _debugcheck=False):
        """
        Description

        Parameters:
            mult_lim:
            iterations(int):
            noise_std(float):
            update(bool):
            _debugcheck(bool):

        Return:
            (ndarray, ndarray): Augmented spectrum and label arrays
        """
        for i in range(iterations):
            # random multiplier(s) generation using uniform distribution
            arr_mult = np.random.uniform(1 - mult_lim, 1 + mult_lim, (self.sp.shape[0], 1))
            # generation of new spectra
            mult_sp = self.multiplier(self.sp, arr_mult)
            if noise_std is not None:
                # noise is added on the new spectra
                mult_sp = self.noise_addition(mult_sp, noise_std)
            # new spectra & labels are appended to the original ones
            self.sp_aug = np.vstack((self.sp_aug, mult_sp))
            self.lab_aug = np.vstack((self.lab_aug, self.lab))

        # the spectra and labels are randomly mixed
        self.sp_aug, self.lab_aug = shuffle(self.sp_aug, self.lab_aug)

        if update:
            # overwrites source data with augmented data
            self.sp = self.sp_aug
            self.lab = self.lab_aug

        if _debugcheck:
            return mult_sp

        return self.sp_aug, self.lab_aug

    def aug_mixup(self, n_spec=2, alpha=0.5, iterations=1, update=False, _debugcheck=False):
        """
        Spectrum linear combinaison process inspired by the Mixup method (Zhang, Hongyi, et al. 2017).

        Parameters:
            n_spec(int): Amount of spectrum mixed together.
            alpha(float): Dirichlet distribution concentration parameter.
            iterations(int): Number of iterations.
            update(bool):
            _debugcheck(bool):

        Returns:
            (ndarray, ndarray): Augmented spectrum and label arrays
        """
        # initialization and space allocation
        alpha_array = np.ones(n_spec) * alpha
        # Lambda values generated with a dirichlet distribution
        lam_s = np.random.dirichlet(alpha_array, self.sp.shape[0])

        for i in range(iterations):
            sp_sum = np.zeros(self.sp.shape)
            lab_sum = np.zeros(self.lab.shape)
            for n in range(0, n_spec):
                arr_, lab_ = shuffle(self.sp, self.lab)
                lam = np.array(lam_s[:, n], ndmin=2).transpose()
                sp_sum = sp_sum + lam * arr_
                lab_sum = lab_sum + lam * lab_

            self.sp_aug = np.vstack((self.sp_aug, sp_sum))
            self.lab_aug = np.vstack((self.lab_aug, lab_sum))

        # the spectra and labels are randomly mixed
        self.sp_aug, self.lab_aug = shuffle(self.sp_aug, self.lab_aug)

        if update:
            # overwrites source data with augmented data
            self.sp = self.sp_aug
            self.lab = self.lab_aug

        if _debugcheck:
            return sp_sum, lab_sum

        return self.sp_aug, self.lab_aug


if __name__ == "__main__":
    print(help(SpecAugPipeline))
