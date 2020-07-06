#  Machine learning and smart techniques for Surface-Enhanced Raman Spectroscopy(SERS) 

This project includes a detailed method to use machine learning with SERS. In addition to being ready to use, fast and efficient, this method stands out through the addition of a number of interesting procedures that have never been presented before for the combination of SERS and machine learning.  The project can be divided into two parts:

* **Data Processing:** Contains all the operations applied on the data (spectra) before using the machine learning models. This includes  y, data augmentation, data preprocessing and data visualization.  

* **Machine Learning models application:** Contains all the steps related to the application of machine learning models on the data resulting from the previous step. This includes model definition, training, validation and testing.

This repository contains all the functions and machine learning models used in the **Lebrun and Boudreau (2020)** published paper.  The trained model and database that were used to obtain the results presented in the article are also available. 

## Getting Started 

This repository main components are the modules **Models.py** and **Function_Repository.py** that contain all the features needed to efficiently use machine learning with SERS. It is strongly suggested to start with the Jupyter notebook script which presents the complete procedure and describes each step in detail while adding information to make it easier to understand.  


**Important:** This project doesn't cover database conception and requires user to have completed this step before using this project. As an indication, the rows of the database must correspond to the different spectra and the columns to the different Raman shift. The column(s) with the labels must be appended to the left of the database.

### Requirements to run the code

Listed below are some of the main modules needed to operate the codes: 

* Keras
* Sklearn
* Scipy
* Numpy
* Pandas
* Matplotlib
* Tensor flow (GPU or CPU)

The complete list with all modules required can be found in the **Requirement.txt** file.

## Principal Features 

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
