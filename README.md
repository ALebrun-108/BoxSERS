# Advanced processing and machine learning for vibrational spectra

This repository includes SpecMaster, a ready-to-use and efficient python library for processing and applying machine learning to vibrational spectra. Two Jupyter notebooks to aid in the use of Specmaster are also included, as well as a pre-trained machine learning model and a database of SERS bile acid spectra that were used in the article published by **Lebrun and Boudreau (2020)**.


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

## Getting Started 

This repository main components are the modules **Models.py** and **Function_Repository.py** that contain all the features needed to efficiently use machine learning with SERS. It is strongly suggested to start with the Jupyter notebook script which presents the complete procedure and describes each step in detail while adding information to make it easier to understand.  


**Important:** This project doesn't cover database conception and requires user to have completed this step before using this project. As an indication, the rows of the database must correspond to the different spectra and the columns to the different Raman shift. The column(s) with the labels must be appended to the left of the database.


SpecMaster package is ready to use and efficient and covers methods for data augmentation, spectral correction, dimensional reduction and data visualization. Some machine training models that offer several options are also included.
## Usage

### Database Splitting
* Generation of the training, validation and test sets
* Visualization feature to check the distribution of the different classes in each newly generated set.

### Data Augmentation
* Spectra mixeup: linear combination of two or three spectra 
* Simple data augmentation methods: Noise addition, offset , multiplicative factor
* Visualization feature to check the results of different data augmentation methods
### Spectra Correction
* Savitsy-Golay Smoothing
* ALS baseline correction 
* Data cut 
### PCA validation
* Principal component analysis visualization 

## Machine Learning Models 
* Convolutional Neural Networt (3 x Convolutional layer 1D , 2 x Dense layer)   


```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```
## License
[MIT](https://choosealicense.com/licenses/mit/)



* **Data Processing:** Contains all the operations applied on the data (spectra) before using the machine learning models. This includes  y, data augmentation, data preprocessing and data visualization.  

* **Machine Learning models application:** Contains all the steps related to the application of machine learning models on the data resulting from the previous step. This includes model definition, training, validation and testing.


