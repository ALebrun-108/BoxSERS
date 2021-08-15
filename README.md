
![test image size](fig/Logo_V5_github.png)
----
**BoxSERS**, a powerful and ready-to-use python package providing several tools for the analysis of 
vibrational spectra (Raman, FTIR, SERS, etc.), including features for data augmentation, 
dimensional reduction, spectral correction and both supervised and unsupervised machine learning.

## General info on the repository
This GitHub repository includes the following elements : 

* **BoxSERS package** : Complete and ready-to-use python library includind  for the application of methods designed and adapted for vibrational spectra(RamSERS, etc.)


* **Jupyter notebooks (soon to be published )** : Typical examples of BoxSERS package usage.
  

* **Raw and preprocessed data (soon to be published)** :  Database of SERS bile acid spectra that were used (Raw and Preprocess form) in the article published by *Lebrun and Boudreau (2021)*. Can be used as a starting point to start using the BoxSERS package.

Below, on this page, is also the package's installation guideline and an overview of its main functions.

## Table of contents
* [Getting Started](#getting-started)
  * [BoxSERS Installation](#boxsers-installation)
  * [Requirements](#requirements)
  * [Label Information](#label-information)
* [Included Features](#included-features)
  * [Module misc_tools](#module-misc_tools)
  * [Module visual_tools](#module-visual_tools)
  * [Module preprocessing](#module-preprocessing)
  * [Module data augmentation](#module-data_augmentation)
  * [Module dimension reduction](#module-dimension_reduction)
  * [Module clustering](#module-clustering)
  * [Module classification](#module-classification)



## Getting Started 

It is advisable to start with the Jupyter notebook  that present the complete procedure and describe each step in detail while adding information to facilitate understanding. 

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

### Label information

The labels associated with the spectra can be either integer values (single column) or binary values (multiple columns).


#### Example of labels for three classes that correspond to three bile acids:

| Bile acid  	| Integer label (1 column) 	| Binary label (3 columns) 	|
|------------------	|:-------------:	|:------------:	|
| Cholic acid          	|       0       	|    [1 0 0]   	|
| Lithocholic acid        	|       1       	|    [0 1 0]   	|
| Deoxycholic acid        	|       2       	|    [0 0 1]   	|

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

from boxsers.preprocessing import als_baseline_cor, spectral_cut, intensity_normalization,spline_interpolation
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

from boxsers.data_augmentation import aug_noise
```

### Module ``dimension_reduction``
This module provides different techniques to perform dimensionality reduction of
vibrational spectra.

* **SpectroPCA** : Principal Component Analysis (PCA) model object.

```python
from boxsers.machine_learning import SpectroPCA
```


### Module ``clustering``
This module provides unsupervised learning models for vibrational spectra cluster analysis.

* **SpectroKmeans** : K-Means clustering model.


* **SpectroGmixture** : Gaussian mixture probability distribution model.

```python
# Code example:

from boxsers.machine_learning import SpectroKmeans, SpectroGmixture
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
