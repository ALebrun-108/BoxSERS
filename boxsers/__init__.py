from boxsers.machine_learning import SpectroGmixture, SpectroKmeans, SpectroRF, SpectroSVM, \
     SpectroLDA, SpectroPCA, SpectroCNN, validation_metrics
from boxsers.data_augmentation import aug_mixup, aug_xshift, aug_noise, aug_multiplier, aug_linslope, aug_offset
from boxsers.preprocessing import als_baseline_cor, savgol_smoothing, spectral_cut, spline_interpolation, \
    spectral_normalization, cosmic_filter
from boxsers.misc_tools import data_split, load_rruff, ramanshift_converter, wavelength_converter
from boxsers.visual_tools import random_plot, spectro_plot, class_plot, distribution_plot


# Version of the boxsers package
__version__ = "1.1.0"
