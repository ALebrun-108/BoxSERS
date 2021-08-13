from boxsers.machine_learning import SpectroGmixture, SpectroKmeans, SpectroRF, SpectroSVM, \
     SpectroLDA
from boxsers.data_augmentation import aug_mixup, aug_xshift, aug_noise, aug_multiplier, aug_linslope, aug_offset
from boxsers.dimension_reduction import SpectroPCA, SpectroFA, SpectroICA
from boxsers.preprocessing import intensity_normalization, spline_interpolation, baseline_subtraction,\
    spectral_cut, savgol_smoothing, median_filter
from boxsers.misc_tools import data_split, database_creator, load_rruff, ramanshift_converter,\
    wavelength_converter, import_sp
from boxsers.visual_tools import random_plot, spectro_plot, class_plot, distribution_plot

# Version of the boxsers package
__version__ = "1.0.0"
