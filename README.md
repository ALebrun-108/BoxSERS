# Advanced processing and machine learning for vibrational spectra
Introduces SpecMaster, a ready-to-use and efficient python library for processing and applying machine learning to vibrational spectra (Raman,FTIR, SERS, etc.). 

![alt text](https://github.com/ALebrun-108/Advanced-processing-and-machine-learning-for-vibrational-spectra/blob/master/beta_single_alpha04.png?raw=true)

## Table of contents
* [General info](#general-info)
* [Setup](#Setup)
* [Usage](#Usage)
* [License](#License)

## General info

This project includes the following elements: 
- SpecMaster package: Covers methods for data augmentation, spectral correction, dimensional reduction and data visualization. Ready-to-use supervised and unsupervised machine learning models with several options are also included in this package.
- Two Jupyter notebooks: Detailed examples of use of the specmaster package.
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


## Usage

### Getting Started 

It is strongly suggested to start with the two Jupyter notebook script which presents the complete procedure and describes each step in detail while adding information to make it easier to understand. 

**Important Notices:** 
- This project doesn't cover database conception and requires user to have completed this step before using this project.
- Labels associated to spectra can either be in the form of integer or binary values.

```python
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# creating instance of labelencoder
labelencoder = LabelEncoder()
# encodes class names as integer labels: [0] [4] [1] ...
data['Classes'] = labelencoder.fit_transform(data['Classes'])

# takes the labels from the database and stores them as a numpy array(int).
lab = data.to_numpy(dtype='int')[0:, 0]
# converts labels to a binary matrix [1 0 0 0] [0 0 0 4] [0 1 0 0]
lab_enc = np_utils.to_categorical(lab)

# takes the spectra from the database and stores them as a numpy array(float64).  
spectra = data.to_numpy(dtype='float64')[0:, 1:]
```

### Spectrum Visualization

Two functions allows to visualize the spectra as graphs 
- **random_plot**: Returns a graph of a certain number of randomly selected spectra.
- **spectro_plot**: Returns a graph of one or more selected spectra.

```python
from specmaster import spectro_plot, random_plot

random_plot(wn, spec, random_spectra=4)  # plots 4 randomly selected spectra
spectro_plot(wn, spec[0])  # plots the first spectrum
spectro_plot(wn, spec[0], spec[2]) # plots first and third spectra
spectro_plot(wn, spec)  # plots all spectra
```

### Database Splitting
Splitting the database spectra into subsets validated by distribution plot.

- **data_split**: Generates two subsets of spectra from the input database.
- **distribution_plot**: Plots the distribution of the different classes in a selected set.

```python
from specmaster import data_split, distribution_plot

# Separates the spectra and the labels into test(0.4) and training(0.6) subsets.
(x_train, x_test, y_train, y_test) = data_split(spectra, labels, b_size=0.4)

# plots the classes distribution within the training set.
distribution_plot(y_train, title='Train set distribution', class_names=classnames)
```

### Spectral Data Augmentation
* Spectra mixeup: linear combination of two or three spectra 
* Simple data augmentation methods: Noise addition, offset , multiplicative factor
* Visualization feature to check the results of different data augmentation methods

```python
from specmaster import SpecAugPipeline as SpecDA 

# determining the average for each spectra contained in the training set
xtrain_mean = np.mean(x_train, axis=1, keepdims=True)

# number of iterations
i = 2  

# creating instance of SpecDA
data_aug_pipeline = SpecDA(x_train, y_train)

# following line corresponds to data augmentation method 
data_aug_pipeline.aug_noise(param_nse*xtrain_mean, iterations=i)
data_aug_pipeline.aug_ioffset(param_ioff*xtrain_mean, iterations=i)
data_aug_pipeline.aug_multiplier(param_mult, iterations=i)
data_aug_pipeline.aug_linear_slope(param_slp*xtrain_mean, param_inter, iterations=i)
data_aug_pipeline.aug_wshift(param_wshft, iterations=i)
data_aug_pipeline.aug_mixup(n_spec=2, alpha=0.40, iterations=3)

# retrieving augmented data
x_train_aug, y_train_aug = data_aug_pipeline.out()
```

### Spectral Data Correction
* Savitsy-Golay Smoothing
* ALS baseline correction 
* Data cut 

### Dimensional reduction
* Principal component analysis visualization

```python
from specmaster.pca_model import SpectroPCA, SpectroFA, SpectroICA

pca_model = SpectroICA(n_comp=50)
pca_model.fit_model(x_train)
pca_model.scatter_plot(x_test, y_test, targets=classnames, comp_x=1, comp_y=2)
pca_model.pca_component(Wn, 2)
x_pca = pca_model.transform_spectra(x_train)

```

### Unsupervised Machine Learning Models 

### Supervised Machine Learning Models 
* Convolutional Neural Networt (3 x Convolutional layer 1D , 2 x Dense layer)   

## License
[MIT](https://choosealicense.com/licenses/mit/)
