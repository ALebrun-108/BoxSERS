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


def spectro_subsampling(sp, lab=None, batch_size=0.5):
    """
    Subsamples a given fraction or number of spectra from an initial spectra array.

    Parameters:
        sp : array
            Input spectra, array shape = (n_spectra, n_pixels)

        lab : array, default=None
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        batch_size : float or integer positive value, default=0.5
             Fraction or number of spectra to sample from the initial spectra array.

    Returns:
        (array) Sampled subset of spectra.

        (array) Sampled subset of labels.
    """
    if 0 < batch_size < 1:
        batch_size = int(batch_size * sp.shape[0])
    else:
        batch_size = batch_size

    random_row = np.random.choice(sp.shape[0], size=batch_size, replace=False)
    sp_sample = sp[random_row, :]
    if lab is not None:
        lab_sample = lab[random_row, :]
        return sp_sample, lab_sample
    return sp_sample


def find_classes_index(lab_array, *labels):
    """
    Returns the indexes corresponding to the desired class(es) (using their corresponding label).

    Parameters:
        lab_array : array
            Labels array. Array shape = (n_spectra,) for integer labels and (n_spectra, n_classes)
            for binary labels.

        *labels : array
            Label(s) coresponding to the desired class(es)

    Returns:
        (array) Indexes of the desired class(es)
    """
    classes_indexes = []
    if lab_array.ndim == 2 and lab_array.shape[1] > 1:  # label_type = 'binary'
        # Find indexes of elements with specific label
        for y in labels:
            indices = np.where(np.all(lab_array == y, axis=1))[0]
            # Extend indexe by appending elements from the indices.
            classes_indexes.extend(indices)

    else:  # label_type = 'int'
        for y in labels:
            indices = np.where(lab_array == y)[0]
            classes_indexes.extend(indices)

    return np.array(classes_indexes)


def remove_classes(sp, lab, *labels_to_remove, print_infos=False):
    """
    Remove desired classes from spectra and labels array

    Parameters:
        sp : array
            Initial set of spectra to split into two new subsets. Array shape = (n_spectra, n_pixels).

        lab : array
            Labels assigned the "sp" spectra. Array shape = (n_spectra,) for integer labels and
            (n_spectra, n_classes) for binary labels.

        *labels_to_remove : array
            Label(s) coresponding to the class(es) to remove.

    Returns:
        (array) Spectra array without the class(es) removed.

        (array) Label array without the class(es) removed.
    """
    # Find indexes of elements with specific label
    indexes = find_classes_index(lab, *labels_to_remove)

    # Deletes spectra & labels belonging to the selected class
    sp_modified = np.delete(sp, indexes, axis=0)
    lab_modified = np.delete(lab, indexes, axis=0)

    if print_infos:
        print('{} spectra & labels removed belonging to {} classes'.format(len(indexes), labels_to_remove))

    return sp_modified, lab_modified


def split_by_classes(sp, lab):
    """
    Divides a spectra array into subsets for each unique label.

    Parameters:
        sp : array
            Initial set of spectra to split into two new subsets. Array shape = (n_spectra, n_pixels).

        lab : array
            Labels assigned the "sp" spectra. Array shape = (n_spectra,) for integer labels and
            (n_spectra, n_classes) for binary labels.

    Returns:
        (dict) Dictionary where keys are unique labels, and the values are the spectra with these labels.
    """
    # Converts binary labels to integer labels. Does nothing if they are already integer labels.
    if lab.ndim == 2 and lab.shape[1] > 1:
        lab = np.argmax(lab, axis=1)

    unique_labels = np.unique(lab)  # Get unique labels
    split_classes_dict = {}  # Dictionary to store subsets

    for u_label in unique_labels:
        mask = (lab == u_label)  # Create a boolean mask for the current label
        split_classes_dict[u_label] = sp[mask]  # Use the mask to extract spectra
    return split_classes_dict


def load_rruff(directory):
    """
    Export a subset of Raman spectra from the RRUFF database in the form of three related lists
    containing Raman shifts, intensities and mineral names.

    Parameters:
        directory : String
            Online directory that leads to a zip file containing the desired set of RRUFF spectra. Here are
            the possible directories for RRUFF Raman spectra:
                - 'https://rruff.info/zipped_data_files/raman/excellent_oriented.zip'
                - 'https://rruff.info/zipped_data_files/raman/excellent_unoriented.zip'
                - 'https://rruff.info/zipped_data_files/raman/fair_oriented.zip'
                - 'https://rruff.info/zipped_data_files/raman/fair_unoriented.zip'
                - 'https://rruff.info/zipped_data_files/raman/poor_oriented.zip'
                - 'https://rruff.info/zipped_data_files/raman/poor_unoriented.zip'

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
