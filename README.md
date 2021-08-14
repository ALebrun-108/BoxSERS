
![test image size](fig/Logo_V3_github.png)
----
**BoxSERS**, a powerful and ready-to-use python package providing several tools for the analysis of 
vibrational spectra (Raman, FTIR, SERS, etc.), including features for data augmentation, 
dimensional reduction, spectral correction and both supervised and unsupervised machine learning.

## General info on the repository
This GitHub repository includes the following elements : 

* **BoxSERS package** : Complete and ready-to-use python library includind  for the application of methods designed and adapted for vibrational spectra(RamSERS, etc.)

* **Jupyter notebooks** : Typical examples of BoxSERS package usage.
  
* **Raw and preprocessed data** :  Database of SERS bile acid spectra that were used (Raw and Preprocess form) in the article published by *Lebrun and Boudreau (2021)*. Can be used as a starting point to start using the BoxSERS package.

Below, on this page, is also the package's installation guideline and an overview of its main functions.

## Table of contents
* [Getting Started](#getting-started)
  * [BoxSERS Installation](#boxsers-installation)
  * [Requirements](#requirements)
* [Included Features](#included-features)
  * [Module misc_tools](#module-misc_tools)
  * [Module visual_tools](#module-visual_tools)
  * [Module preprocessing](#module-preprocessing)
  * [Module data augmentation](#module-data_augmentation)
  * [Module dimension reduction](#module-dimension_reduction)
  * [Module clustering](#module-clustering)
  * [Module classification](#module-classification)
* [Label Information](#label-information)


## Getting Started 

It is strongly suggested to start with the two Jupyter notebook script which presents the complete 
procedure and describes each step in detail while adding information to make it easier to understand. 

This project doesn't cover database conception and requires user to have
completed this step before using this project.

### BoxSERS Installation

From PypY
```bash
pip install boxsers
```

From Github 
```bash
pip install git+https://github.com/ALebrun-108/BoxSERS.git
```

### Requirements
Listed below are the main modules needed to operate the codes: 

* Sklearn
* Scipy
* Numpy
* Pandas
* Matplotlib
* Tensor flow (GPU or CPU)


## Included Features
This section includes the detailed description (utility, parameters, ...) for each function and
class contained in the BoxSERS package
___

### Module ``misc_tools``
This module provides functions for a variety of utilities.

* **data_split** : Randomly splits an initial set of spectra into two new subsets named in this
  function: subset A and subset B.


* **ramanshift_converter** : Converts wavelength [nm] to Raman shifts [cm-1].


* **wavelength_converter** : Convert Raman shifts [cm-1] to wavelengths [nm].


* **load_rruff** : Export a subset of Raman spectra from the RRUFF database in the form of three related lists
  containing Raman shifts, intensities and mineral names.



### Module ``visual_tools``
This module provides different tools to visualize vibrational spectra quickly.

* **spectro_plot :** Returns a plot with the selected spectrum(s)


* **random_plot :** Plot a number of randomly selected spectra from a set of spectra.


* **distribution_plot :** Return a bar plot that represents the distributions of spectra for each classes in
  a given set of spectra

```python
# Code example:

from boxsers.usefultools import data_split, distribution_plot

# randomly splits the spectra(spec) and the labels(lab) into test and training subsets.
(spec_train, spec_test, lab_train, lab_test) = data_split(spec, lab, test_size=0.4)  
# resulting train|test set proportions = 0.6|0.4


# plots the classes distribution within the training set.
distribution_plot(lab_train, title='Train set distribution')
```
![test image size](fig/distribution.png)

```python
# Code example:

from boxsers.visual_tools import spectro_plot, random_plot

# spectra array = spec, raman shift column = wn
random_plot(wn, spec, random_spectra=4)  # plots 4 randomly selected spectra
spectro_plot(wn, spec[0], spec[2])  # plots first and third spectra
```
![test image size](fig/random5_plot.png)


### Module ``preprocessing``
This module provides functions to preprocess vibrational spectra. These features
improve spectrum quality and can improve performance for machine learning applications.

* **als_baseline_cor** : Subtracts the baseline signal from the spectrum(s) using an
  Asymmetric Least Squares estimation.


* **intensity_normalization** : Normalizes the spectrum(s) using one of the available norms in this function.


* **savgol_smoothing** : Smoothes the spectrum(s) using a Savitzky-Golay polynomial filter.


* **cosmic_filter** : Applies a median filter to the spectrum(s) to remove cosmic rays.


* **spectral_cut** : Subtracts or sets to zero a delimited spectral region of the spectrum(s).


* **spline_interpolation** : Performs a one-dimensional interpolation spline on the spectra to reproduce
  them with a new x-axis.

```python
# Code example:

from boxsers.preprotools import baseline_subtraction, spectral_cut, spectral_normalization, spline_interpolation

# interpolates with splines the spectra and converts them to a new raman shift range(new_wn)
new_wn = np.linspace(500, 3000, 1000)
spec_cor = spline_interpolation(spec, wn, new_wn)
# removes the baseline signal measured with the als method 
(spec_cor, baseline) = baseline_subtraction(spec, lam=1e4, p=0.001, niter=10)
# normalizes each spectrum individually so that the maximum value equals one and the minimum value zero 
spec_cor = spectral_normalization(spec)
# removes part of the spectra delimited by the Raman shift values wn_start and wn_end 
spec_cor, wn_cor = spectral_cut(spec, wn, wn_start, wn_end)
```
![test image size](fig/correction.png)


### Module ``data_augmentation``
This module provides funtions to generate new spectra by adding different variations to
existing spectra.

* **aug_mixup** : Randomly generates new spectra by mixing together several spectra with a Dirichlet
  probability distribution.


* **aug_noise** : Randomly generates new spectra with Gaussian noise added.


* **aug_multiplier** : Randomly generates new spectra with multiplicative factors applied.


* **aug_offset** : Randomly generates new spectra shifted in intensity.


* **aug_xshift** : Randomly generates new spectra shifted in wavelength.


* **aug_linslope** : Randomly generates new spectra with additional linear slopes

```python
# Code example:

from boxsers.dataugtools import SpectroDataAug

spec_nse, _  = SpectroDataAug.aug_noise(spec, lab, param_nse, mode='check')
spec_mult_sup, _ = SpectroDataAug.aug_multiplier(spec, lab, 1+param_mult, mode='check')
spec_mult_inf, _ = SpectroDataAug.aug_multiplier(spec, lab, 1-param_mult, mode='check')

legend = ['initial', 'noisy', 'multiplier superior', 'multiplier inferior']
spectro_plot(wn, spec, spec_nse, spec_mult_sup, spec_mult_inf, legend=legend)

spec_nse, lab_nse = SpectroDataAug.aug_noise(spec, lab, param_nse, quantity=2, mode='random')
spec_mul, lab_mul = SpectroDataAug.aug_multiplier(spec, lab, mult_lim, quantity=2, mode='random')

# stacks all generated spectra and originals in a single array
spec_aug = np.vstack((x, spec_nse, spec_mul))
lab_aug = np.vstack((lab, lab_nse, lab_mul))

# spectra and labels are randomly mixed
x_aug, y_aug = shuffle(x_aug, y_aug)
```

### Module ``dimension_reduction``
This module provides different techniques to perform dimensionality reduction of
vibrational spectra.

* **SpectroPCA** : Principal Component Analysis (PCA) model object.

```python
from boxsers.pca_model import SpectroPCA, SpectroFA, SpectroICA

pca_model = SpectroICA(n_comp=50)
pca_model.fit_model(x_train)
pca_model.scatter_plot(x_test, y_test, targets=classnames, comp_x=1, comp_y=2)
pca_model.pca_component(Wn, 2)
x_pca = pca_model.transform_spectra(x_train)
```
![test image size](fig/data_reduce.png)


### Module ``clustering``
This module provides unsupervised learning models for vibrational spectra cluster analysis.

* **SpectroKmeans** : K-Means clustering model.


* **SpectroGmixture** : Gaussian mixture probability distribution model.

```python
# Code example:

from boxsers.machine_learning import SpectroGmixture, SpectroKmeans

kmeans_model = SpectroKmeans(n_cluster=5)
kmeans_model.fit_model(spec_train)
kmeans_model.scatter_plot(spec_test)
```

### Module ``classification``
This module provides supervised learning models for vibrational spectra classification.

* **SpectroRF** :  Random forest classification model.


* **SpectroSVM** : Support Vector Machine classification model.


* **SpectroLDA** : Linear Discriminant Analysis classification model


### Module ``neural_networks``
This module provides neural network model specifically designed for the
classification of vibrational spectra.

* **SpectroCNN** : Convolutional Neural Network (CNN) for vibrational spectra classification.

## Label information

Labels associated to spectra can be in one of the following three forms:

| Label Type    | Examples                             |
| ------------- | ------------------------------------ |
| Text          | Cholic, Deoxycholic, Lithocholic, ...|
| Integer       | 0, 3, 1 , ...                        |
| Binary        | [1 0 0 0], [0 0 0 1], [0 1 0 0], ... |

Function/Class | Interger Label | Binary label | Continuous Label
:------------ | :-------------| :-------------| :-------------
`data_split` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
`distribution_plot` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: