"""
Author: Alexis Lebrun (Master's student)
School: Université Laval

The "ramanbox.useful_features" module provides functions for various utilities.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import os
import re
import tempfile
logging.basicConfig(format='%(levelname)s:%(message)s')


def remove_class(df, label):
    """ Removes a specific label from a spectrum database.

    Parameters:
        df : pandas dataframe
            Database with the spectra and their associated labels.

        label : string(class names) or integer(label)
            Label of the class to remove from the spectrum database.

    Returns:
        (pandas dataframe) Database following the subtraction of a specific label.
    """

    # removes all spectra associated with the label and resets the indexes
    df = df[df.Classes != label].reset_index(drop=True)
    # the list of class names (or label) is updated
    new_classnames = df.Classes.unique()
    return df, new_classnames


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


def distribution_plot(lab, bar_width=0.8, avg_line=False, class_names=None,  title=None,  ylabel='Number of samples',
                      xlabel=None, color='orange', fontsize=10, fig_width=6.08, fig_height=3.8,
                      save_path=None):
    """
    Return a bar plot that represents the distributions of spectra for each classes in
    a given set of spectra.

    Parameters:
        lab : array
            Labels assigned the "sp" spectra. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        bar_width : positive float between 0 and 1, default= 0.8
            Bar width.

        avg_line : boolean, default=False
            If True, add a horizontal line corresponding to the average number of spectra per class.

        class_names : list or tupple, default=None
            Names or labels associated to the classes. If None, classes are in order, but their names
            are not displayed.

        title : string, default=None
            Plot title. If None, there is no title displayed.

        ylabel : string, default='Number of samples'
            Y-axis title. If None, there is no title displayed.

        xlabel : string, default=None
            X-axis title. If None, there is no title displayed.

        color : string, default='orange'
            Bar color.

        fontsize : positive float, default= 10
            Font size used for the different elements of the graph.

        fig_width : positive float or int, default=6.08
            Figure width in inches.

        fig_height : positive float or int, default=3.8
            Figure height in inches.

        save_path : string, default=None
            Path where the figure is saved. If None, saving does not occur.

    Return:
        Bar plot that shows the distribution of spectra in each class.
    """
    # Converts binary labels to integer labels. Does nothing if they are already integer labels.
    if lab.ndim == 2 and lab.shape[1] > 1:  # lab is a binary matrix (one-hot encoded label)
        lab = np.argmax(lab, axis=1)

    unique, distribution = np.unique(lab, return_counts=True, axis=None)
    mean = distribution.mean()
    y_pos = np.arange(len(distribution))  # bar tick positions

    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)
    # plots the distribution barplot
    ax.bar(y_pos, distribution, bar_width, color=color, edgecolor='black')
    if avg_line is True:
        # Average horizontal line coordinates
        x_coordinates = [-0.6 * bar_width, y_pos[-1] + 0.6 * bar_width]
        y_coordinates = [mean, mean]
        # Draws a vertical line indicating the average number of spectra per class
        ax.plot(x_coordinates, y_coordinates, ls='--', lw=2, label='Average spectra/class')
        ax.legend(loc='best', fontsize=fontsize-2)  # 2.0 points smaller font size
    # title settings
    ax.set_title(title, fontsize=fontsize + 1.2)  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=fontsize)  # sets the x-axis title
    ax.set_ylabel(ylabel, fontsize=fontsize)  # sets the y-axis title
    # axis limit settings
    ax.set_xlim(-0.6 * bar_width, y_pos[-1] + 0.6 * bar_width)
    ax.set_ylim(0, 1.1 * np.max(distribution))
    # tick settings
    if class_names is not None:
        ax.set_xticks(y_pos)
        ax.set_xticklabels(class_names, fontsize=fontsize)
    # adjusts subplot params so that the subplot(s) fits in to the figure area
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


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
    Converts wavelength[nm] to Raman shifts[cm-1].

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
    Convert Raman shifts[cm-1] to wavelengths[nm].

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


def database_creator(directory, class_names=None, nfiles_class=None, checkorder=False, skiprows=2):
    """
    Returns a database (pandas dataframe) containing the spectra and the associated labels, along
    with the array of x-axis values (Raman shift, wavelength, etc.) associated with the spectra.
    TODO: changer creator pour generator

    Notes:
        Intended to be used only on the text files produced by the Raman spectrometer (Dboudreau)

        Parameter nfiles_class is taken into account only if class_names contains more than one class.

        The alphabetical order of the .txt files in the directory must be followed for the lists
        or tupples class_names and nfiles_class.

        Files in the directory must have the same wavelength calibration.

        Files must follow the following format:
            -First column = wavelenght or Raman shift
            -Other columns = one spectra per column
            -Apart from the files produced during the covid-19 crash, the first two rows at the top
             correspond to the hyperspectral coordinates in x and y

    Parameters:
        directory : string
            Directory (path) of the spectra files(.txt).

        class_names : string, integer, list or tupple, default=None
            Names associated to the classes present in the database.

        nfiles_class : list or tupple, default=None
            Number of (.txt) files in "directory" for each class contained in "class_names".

        checkorder : boolean, default=False
            If true, print the file names in the order used to build the database

        skiprows : integer, default=2
            Number of rows in .txt files to skip. With some exceptions (covid 19 crash), the
            text files exported from the spectroRamanX always include two rows for hyperspectral
            coordinates that need to be removed, which explains the default value of 2.

    Returns:
        (pandas dataframe) Database with the spectra and their associated labels.

        (array) X-axis(wavenumber, wavelenght, Raman shift, etc.) used for the spectra.
                Array shape = (n_pixels, ).
    """
    # labels list space allocation
    labels = []
    # retrieves all text files in the given directory
    filenames = [f for f in os.listdir(directory) if f.endswith(".txt")]
    filenames.sort()  # sort files in alphabetical order

    if checkorder is True:
        print(filenames)

    if class_names is None:
        # no class name is given, labels are set to "unknown" for all spectra
        labels = ['unknown'] * len(filenames)
    elif isinstance(class_names, (str, int)):
        # all spectra belong to the same class, the same label is used for all spectra.
        labels = [class_names] * len(filenames)
    elif isinstance(class_names, (list, tuple)):
        # different classes are used, different labels are given to spectra files
        if len(class_names) == len(nfiles_class):
            for i in range(len(class_names)):
                x = [class_names[i]] * nfiles_class[i]
                labels = labels + x
        else:
            raise ValueError('if class_names is a list or a tupple, its number of elements must correspond'
                             'to the number of elements in nfiles_classes')
    # space allocation for
    dataframe = pd.DataFrame()
    wn = []

    for (name, lab) in zip(filenames, labels):
        df = pd.read_csv(directory + name, header=None, decimal='.', sep=',', skiprows=skiprows, comment='#')
        df = df.T
        wn = df.to_numpy(dtype='float64')[0, 0:]
        df.drop(df.index[0], inplace=True)  # Retire les index et les nombres d'ondes
        df.insert(0, 'Classes', lab)  # Spécifie les labels pour chaques échantillons
        dataframe = pd.concat([dataframe, df], ignore_index=True)
    return dataframe, wn


def import_sp(directory, skiprows=2):
    """
    Returns a database (pandas dataframe) containing the spectra and the associated labels, along
    with the array of x-axis values (Raman shift, wavelength, etc.) associated with the spectra.

    Notes:
        Intended to be used only on the text files produced by the Raman spectrometer (Dboudreau)

    Parameters:
        directory : string
            Directory of the folder that contains the spectra files(.txt).

        skiprows : integer, default=2
            Number of rows in .txt files to skip. With some exceptions (covid 19 crash), the
            text files exported from the spectroRamanX always include two rows for hyperspectral
            coordinates that need to be removed, which explains the default value of 2.

    Returns:
        (array) Extracted spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
                for a single spectrum.

        (array) X-axis(wavenumber, wavelenght, Raman shift, etc.) used for the spectra.
                Array shape = (n_pixels, ).
    """

    #  First column corresponds to the Raman shift or wavelength.
    wn = np.loadtxt(directory, delimiter=',', skiprows=skiprows)[:, 0]
    #  Other columns contain the intensity values of the spectra
    sp = np.loadtxt(directory, delimiter=',', skiprows=skiprows)[:, 1:]

    return sp, wn


if __name__ == "__main__":
    help(__name__)
