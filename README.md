# Advanced processing and machine learning for vibrational spectra
Introduces **SpecMaster**, a complete and ready-to-use python library for the application of data augmentation, dimensional reduction, spectral correction, machine learning and other methods specially designed and adapted for vibrational spectra(Raman,FTIR, SERS, etc.). 

![alt text](https://github.com/ALebrun-108/Advanced-processing-and-machine-learning-for-vibrational-spectra/blob/master/beta_single_alpha04.png?raw=true)

## Table of contents
* [General info](#general-info)
* [Setup](#Setup)
* [Features](#Features)
  * [Spectrum Visualization](#Spectrum-Visualization)
  * [Database Splitting](#Database-Splitting)
  * [Data Augmentation](#Data-Augmentation)
  * [Spectral Correction](#Spectral-Correction)
  * [Dimensional Reduction](#Dimensional-Reduction)
  * [Unsupervised Machine Learning](#Unsupervised-Machine-Learning)
  * [Supervised Machine Learning](#Supervised-Machine-Learning) 
* [License](#License)

## General info

This project includes the following elements: 
- SpecMaster package: Covers methods for data augmentation, spectral correction, dimensional reduction and data visualization. Ready-to-use supervised and unsupervised machine learning models with several options are also included in this package.
- Two Jupyter notebooks: Detailed examples of use of the specmaster package.
  - Data treatment: 
  - Machine learning application: 
- A pre-trained machine learning model and a database of SERS bile acid spectra that were used in the article published by **Lebrun and Boudreau (2020)** and that can be used as a starting point to start using the specmaster package.

## Setup

### SpecMaster Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install specmaster.

```bash
pip install specmaster
```

### Requirements
Listed below are the main modules needed to operate the codes: 

* Keras
* Sklearn
* Scipy
* Numpy
* Pandas
* Matplotlib
* Tensor flow (GPU or CPU)


## Features

### Getting Started 

It is strongly suggested to start with the two Jupyter notebook script which presents the complete procedure and describes each step in detail while adding information to make it easier to understand. 

This project doesn't cover database conception and requires user to have completed this step before using this project.

Labels associated to spectra can be in one of the following three forms:

| Label Type    | Examples                             |
| ------------- | ------------------------------------ |
| Binary        | [1 0 0 0], [0 0 0 1], [0 1 0 0], ... |
| Integer       | 0, 3, 1 , ...                        |
| Text          | sample_0, sample_3, sample_1, ...    |

### Spectrum Visualization

Fast and simple visualization of spectra as graphs 
- **random_plot**: Returns a graph of a certain number of randomly selected spectra.
- **spectro_plot**: Returns a graph of one or more selected spectra.

```python
# Code example:

from specmaster.useful_features import  spectro_plot, random_plot

# spectra array = spec, raman shift column = wn
random_plot(wn, spec, random_spectra=4)  # plots 4 randomly selected spectra
spectro_plot(wn, spec[0])  # plots the first spectrum
spectro_plot(wn, spec[0], spec[2]) # plots first and third spectra
spectro_plot(wn, spec)  # plots all spectra
```

### Database Splitting
Splitting the database spectra into subsets that can be validated using distribution plot.

- **data_split**: Generates two subsets of spectra from the input database.
- **distribution_plot**: Plots the distribution of the different classes in a selected set.

```python
# Code example:

from specmaster.useful_features import data_split, distribution_plot

# randomly splits the spectra(spec) and the labels(lab) into test and training subsets.
(spec_train, spec_test, lab_train, lab_test) = data_split(spec, lab, test_size=0.4, report_enabled=True)  
# resulting train|test set proportions = 0.6|0.4
# report_enabled=True print a distribution report 

# plots the classes distribution within the training set.
distribution_plot(lab_train, title='Train set distribution')
```
![alt text](https://github.com/ALebrun-108/Advanced-processing-and-machine-learning-for-vibrational-spectra/blob/master/distribution.png?raw=true)
### Data Augmentation
* Spectra mixeup: linear combination of two or three spectra 
* Simple data augmentation methods: Noise addition, offset , multiplicative factor
* Visualization feature to check the results of different data augmentation methods

```python
# Code example:

from specmaster.augmentation import SpecAugPipeline 

# creating instance of SpecAugPipeline that is applied the spectra(spec) and the labels(lab)
data_aug_pipeline = SpecAugPipeline(spec, lab)

# following line corresponds to data augmentation method 
data_aug_pipeline.aug_noise(noise_standard_deviation, iterations=2)
data_aug_pipeline.aug_ioffset(intensity_offset_limits, iterations=2)
data_aug_pipeline.aug_multiplier(multiplier_factor_limits, iterations=2)
data_aug_pipeline.aug_linear_slope(slope_limits, intercept_limits, iterations=2)
data_aug_pipeline.aug_wshift(param_wshft, iterations=2)
data_aug_pipeline.aug_mixup(n_spec=2, alpha=0.40, iterations=2)

# retrieving augmented data
spec_aug, lab_aug = data_aug_pipeline.out()
```

### Spectral Correction
* Savitsy-Golay Smoothing
* ALS baseline correction 
* Data cut 
```python
# Code example:

from specmaster import baseline_subtraction, spectral_cut, spectral_normalization, spline_interpolation

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

### Dimensional Reduction
- **SpectroPCA**: Principal component analysis model
- **SpectroFA**: Factor analysis model
- **SpectroICA**: Independant component analysis model

```python
# Code example:

from specmaster.dim_reduction import SpectroPCA, SpectroFA, SpectroICA

pca_model = SpectroPCA(n_comp=50)
pca_model.fit_model(spec_train)
pca_model.scatter_plot(spec_test, spec_test, targets=classnames, component_x=1, component_y=2)
pca_model.component_plot(wn, component=2)
spec_pca = pca_model.transform_spectra(spec_test)
```

### Unsupervised Machine Learning 
```python
# Code example:

from specmaster.machine_learning import SpectroGmixture, SpectroKmeans

kmeans_model = SpectroKmeans(n_cluster=5)
kmeans_model.fit_model(spec_train)
kmeans_model.scatter_plot(spec_test)
```

### Supervised Machine Learning 
* Convolutional Neural Networt (3 x Convolutional layer 1D , 2 x Dense layer) 
```python
from specmaster.pca_model import SpectroPCA, SpectroFA, SpectroICA

pca_model = SpectroICA(n_comp=50)
pca_model.fit_model(x_train)
pca_model.scatter_plot(x_test, y_test, targets=classnames, comp_x=1, comp_y=2)
pca_model.pca_component(Wn, 2)
x_pca = pca_model.transform_spectra(x_train)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
