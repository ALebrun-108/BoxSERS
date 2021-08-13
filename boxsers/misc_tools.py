"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides functions for a variety of utilities.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import os
import re
import tempfile
logging.basicConfig(format='%(levelname)s:%(message)s')


def data_split(sp, lab, b_size, rdm_ste=None, print_report=False):
    """
    Randomly splits an initial set of spectra into two new subsets named in this
    function: subset A and subset B.

    Parameters:
        sp : array
            Initial set of spectra to split into two new subsets. Array shape = (n_spectra, n_pixels).

        lab : array
            Labels assigned the "sp" spectra. Array shape = (n_spectra,) for integer labels and
            (n_spectra, n_classes) for binary labels.

        b_size : positive float value between 0 and 1
            Ratio of the number of spectra assigned to subset B to the initial number of
            spectra in "sp".

        rdm_ste : integer, default=None
            Random seed of the split. Using the same seed results in the same subsets
            each time.

        print_report : boolean, default=False
            If True, print a distribution report.

    Returns:
        (array) Subset A spectra. Array shape = (n_spectra_in_A, n_pixels).

        (array) Subset B spectra. Array shape = (n_spectra_in_B, n_pixels).

        (array) Subset A labels. Array shape = (n_spectra_in_A, ) for integer labels and (n_spectra_in_A, n_classes)
                for binary label.

        (array) Subset B labels. Array shape = (n_spectra_in_B, ) for integer labels and (n_spectra_in_A, n_classes)
                for binary label.
    """
    # split into two subsets of spectra
    (sp_a, sp_b, lab_a, lab_b) = train_test_split(sp, lab, test_size=b_size, random_state=rdm_ste)  # from sklearn

    if print_report is True:
        # Converts binary labels to integer labels. Does nothing if they are already integer labels.
        if lab.ndim == 2 and lab.shape[1] > 1:  # lab is a binary matrix
            lab_a_dist = np.argmax(lab_a, axis=1)
            lab_b_dist = np.argmax(lab_b, axis=1)
        else:
            lab_a_dist = lab_a
            lab_b_dist = lab_b

        # distribution of the different classes within the two sets.
        distribution_a = np.unique(lab_a_dist, return_counts=True, axis=None)[1]
        distribution_b = np.unique(lab_b_dist, return_counts=True, axis=None)[1]
        print("\nSubset A shape :", sp_a.shape)
        print("Subset A distribution  : ", distribution_a)
        print("\nSubset B shape :", sp_b.shape)
        print("Subset B distribution  : ", distribution_b, '\n')

    return sp_a, sp_b, lab_a, lab_b


def load_rruff(directory):
    """
    Export a subset of Raman spectra from the RRUFF database in the form of three related lists
    containing Raman shifts, intensities and mineral names.

    Parameters:
        directory : String
            Online directory that leads to a zip file containing the desired set of RRUFF spectra. Here are
            the possible directories for RRUFF Raman spectra:
                - 'http://rruff.info/zipped_data_files/raman/excellent_oriented.zip'
                - 'http://rruff.info/zipped_data_files/raman/excellent_unoriented.zip'
                - 'http://rruff.info/zipped_data_files/raman/fair_oriented.zip'
                - 'http://rruff.info/zipped_data_files/raman/fair_unoriented.zip'
                - 'http://rruff.info/zipped_data_files/raman/poor_oriented.zip'
                - 'http://rruff.info/zipped_data_files/raman/poor_unoriented.zip'

    Return:
        (List of numpy array) List that contains Raman shift arrays.

        (List of numpy array) List that contains intensity arrays.

        (List of string) List that contains RRUFF labels (mineral names).
    """
    resp = urlopen(directory)  # opens the online directory
    # creates a temporary directory that will be cleared at the end of this function
    temp_dir = tempfile.TemporaryDirectory()

    with ZipFile(BytesIO(resp.read()), 'r') as zipfile:
        # extracting all the RRUFF files
        print('Extracting all the files now...')
        zipfile.extractall(path=temp_dir.name)
        print('Done!')
        arr_txt = [x for x in os.listdir(temp_dir.name) if x.endswith(".txt")]
        arr_txt.sort()
        # space allocation
        label_list = []
        wn_list = []
        sp_list = []
        n_file_skipped = 0
        for file in arr_txt:
            try:
                label = re.search('(.+?)__', file).group(1)
                df = pd.read_csv(temp_dir.name + '/' + file, header=None, usecols=range(0, 2),
                                 delimiter=',', skiprows=10)
                df.drop(df.tail(1).index, inplace=True)
                wn = df.to_numpy(dtype='float')[:, 0]
                sp = df.to_numpy(dtype='float')[:, 1]
                label_list.append(label)
                wn_list.append(wn)
                sp_list.append(sp)
            except ValueError:
                n_file_skipped += 1
        print("\nDataset #spectra: ", len(sp_list))
        print("Dataset #species: ", len(set(label_list)))
        print("Skipped: ", n_file_skipped, "files")
        temp_dir.cleanup()  # remove the temporary directory
        return wn_list, sp_list, label_list


def ramanshift_converter(x, wl):
    """
    Converts wavelength [nm] to Raman shifts [cm-1].

    Parameters:
        x : array
            X-axis(wavelengths in nm) used for the spectra. Array shape = (n_pixels, ).

        wl : Int or float
            excitation wavelength (in nm) of the laser

    Returns:
        (array) New x-axis in raman shift [cm-1].
                Array shape = (n_pixels, ).
    """
    raman_shift = (1/wl - 1/x)*1E7

    return raman_shift


def wavelength_converter(x, wl):
    """
    Convert Raman shifts [cm-1] to wavelengths [nm].

    Parameters:
        x : array
            X-axis(Raman shift in cm-1) used for the spectra. Array shape = (n_pixels, ).

        wl : Int or float
            excitation wavelength (in nm) of the laser

    Returns:
        (array) New x-axis in wavelength [nm].
                Array shape = (n_pixels, ).
    """
    wavelength = 1/(1/wl-x/1E7)
    return wavelength


if __name__ == "__main__":
    help(__name__)
