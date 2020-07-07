import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import pandas as pd
logging.basicConfig(format='%(levelname)s:%(message)s')


def data_split(sp, lab, b_size=0.3, rdm_ste=None, report_enabled=False):
    """
    Split the database into two set.

    Takes the test_train_split function from scikit-learn and adds additional information about
    the two subsets that are generated

    Parameters:
        sp: Input spectrum array(spectra/row).
        lab: Label associated to sp.
        b_size: Size of subset b.
        rdm_ste(int): Random seed of the split
        report_enabled(bool): print a distribution report

    Returns:
        sp_a, sp_b, lab_a, lab_b: spectra and labels of subsets A and B
    """

    (sp_a, sp_b, lab_a, lab_b) = train_test_split(sp, lab, test_size=b_size, random_state=rdm_ste)

    # Distribution of the different classes within the two sets.
    if report_enabled:

        if lab.ndim == 2 and lab.shape[1] > 1:  # lab is a binary matrix (one-hot encoded label)
            lab_a_dist = np.argmax(lab_a, axis=1)
            lab_b_dist = np.argmax(lab_b, axis=1)
        else:  # lab is a one-column vector with interger value
            lab_a_dist = lab_a
            lab_b_dist = lab_b

        distribution_a = np.unique(lab_a_dist, return_counts=True, axis=None)[1]
        distribution_b = np.unique(lab_b_dist, return_counts=True, axis=None)[1]
        print("\nSubset A shape :", sp_a.shape)
        print("Subset A distribution  : ", distribution_a)
        print("\nSubset B shape :", sp_b.shape)
        print("Subset B distribution  : ", distribution_b, '\n')

    return sp_a, sp_b, lab_a, lab_b


def distribution_plot(lab, class_names=None, save_name=None, **plt_opt):
    """
    Return a bar plot that represents the class distributions in a set

    Parameters:
        lab: labels array
        class_names: Names associated to the class
        save_name: example(name.png, spectro.svg , calcium.jpg, etc.). If None(default), saving does not occur

    Keyword arguments:
        title: Plot title.
        ylabel: Y-axis label.
        line_w: Line width.
        col: Line color(s).
    """

    if lab.ndim == 2 and lab.shape[1] > 1:  # lab is a binary matrix (one-hot encoded label)
        lab = np.argmax(lab, axis=1)
    else:  # lab is a one-column vector with interger value
        lab = lab

    title = plt_opt.get('title', None)
    ylabel = plt_opt.get('ylabel', 'counts')
    line_w = plt_opt.get('line_w', 1)
    col = plt_opt.get('col', 'orange')

    distribution = np.unique(lab, return_counts=True, axis=None)[1]
    mean = distribution.mean()
    w = 0.8  # bar width
    y_pos = np.arange(len(distribution))  # bar tick positions
    # Average horizontal line coordinates
    x_coordinates = [-0.5*w, y_pos[-1] + 0.5*w]
    y_coordinates = [mean, mean]

    fig = plt.figure()
    with plt.style.context('seaborn-notebook'):
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(y_pos, distribution, w, color=col, edgecolor='black', linewidth=line_w)
        ax.plot(x_coordinates, y_coordinates, ls='--', label='Class size average')
        ax.set_xticks(y_pos)
        ax.set_xlim(-0.6 * w, y_pos[-1] + 0.6 * w)
        if class_names is not None:
            ax.set_xticklabels(class_names, rotation=90)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(loc='lower right')
        fig.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, dpi=300)
    return plt.show()


def random_plot(w, sp, random_spectra, save_name=None, **plt_opt):
    """
    Returns a graph of randomly selected spectra

    Parameters:
        w: X-axis values, example(raman shift, wavenumber, wavelength, etc.).
        sp: Input spectrum or spectrum array(spectra/row).
        random_spectra(int): Defines the amount of spectrum that are randomly selected from sp to be plotted.
        save_name: example(name.png, spectro.svg , calcium.jpg, etc.). If None(default), saving does not occur.

    Keyword arguments:
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        line_w: Line width.
        line_s: Line style.
        col: Line color(s).
        legend(list or tuple): List or tuple of legend labels.
        grid(bool): Grid On(True, default) or Off(False).
  """

    if random_spectra > 0:
        if sp.ndim > 1:  # when this condition is not true, the parameter random_spectra is not taken into account
            if sp.shape[0] > random_spectra:
                random_row = np.random.choice(sp.shape[0], size=random_spectra, replace=False)
                sp = sp[random_row, :]
            else:
                raise ValueError('random_spectra size exceeded the number of samples in spectrum dataset')
    else:
        raise ValueError('random_spectra must be an integer greater than zero')

    # .get('key',default): Return the value for key if key is in the dictionary, else default.
    title = plt_opt.get('title', None)
    xlabel = plt_opt.get('xlabel', 'Raman Shift (cm$^{-1}$)')
    ylabel = plt_opt.get('ylabel', None)
    line_w = plt_opt.get('line_w', 1.5)
    line_s = plt_opt.get('line_s', None)
    col = plt_opt.get('col', None)
    legend = plt_opt.get('legend', ())
    grid = plt_opt.get('grid', True)

    fig = plt.figure()
    with plt.style.context('seaborn-notebook'):
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(w, sp.T, color=col, lw=line_w, ls=line_s)
        if grid:
            ax.grid()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(legend)
        fig.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
    return plt.show()


def spectro_plot(w, *sp, save_name=None, **plt_opt):
    """
    Returns a plot of the selected spectrum(s)

    Parameters:
        w: X-axis values, example(raman shift, wavenumber, wavelength, etc.).
        sp: Input spectrum/spectra or spectrum array(s)(spectra/row).
        save_name: example(name.png, spectro.svg , calcium.jpg, etc.). If None(default), saving does not occur.

    Keyword arguments:
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        line_w: Line width.
        line_s: Line style.
        col: Line color(s).
        legend(list or tuple): List or tuple of legend labels.
        grid(bool): Grid On(True, default) or Off(False).
    """
    # .get('key',default): Return the value for key if key is in the dictionary, else default.
    title = plt_opt.get('title', None)
    xlabel = plt_opt.get('xlabel', 'Raman Shift (cm$^{-1}$)')
    ylabel = plt_opt.get('ylabel', None)
    line_w = plt_opt.get('line_w', 1.5)
    line_s = plt_opt.get('line_s', None)
    col = plt_opt.get('col', None)
    legend = plt_opt.get('legend', ())
    grid = plt_opt.get('grid', True)

    fig = plt.figure()
    with plt.style.context('seaborn-notebook'):
        ax = fig.add_subplot(1, 1, 1)
        for s in sp:
            ax.plot(w, s.T, color=col, lw=line_w, ls=line_s)
        if grid:
            ax.grid()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(legend)
        fig.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
    return plt.show()


def database_creator(files_dir, file_names, class_names=None, nfiles_class=None):
    """
    Returns a database(pandas dataframe) and the associated wavelength or Raman shift vector

    Parameters:
        files_dir(str): Directory of the folder that contains the SERS spectra files(.txt)
        file_names(list): Names of SERS(.txt) files contained in file_dir.
        class_names(list): Name of the classes present in the database
        nfiles_class(list): Number of files in file_names for each class contained in class_names.

    Notes:
        - Files must follow the following format:
            first column = wavelenght or Raman shift
            other columns = one spectra per column

        - Files in the directory must have the same wavelength calibration.
        - Files order in file_names must be followed to define class_names and nfiles_class.
        - Parameter nfiles_class is taken into account only if class_names contains more than one class.
    """
    labels = []
    if class_names is None:
        labels = ['unknown']*len(file_names)
    elif len(class_names) == 1:
        labels = class_names * len(file_names)
    elif len(class_names) > 1:
        if len(class_names) == len(nfiles_class):
            for i in range(len(class_names)):
                x = [class_names[i]] * nfiles_class[i]
                labels = labels + x
        else:
            raise ValueError('class_names and nfiles_class must be the same size')

    data = pd.DataFrame()
    wn = []

    for (name, lab) in zip(file_names, labels):
        df = pd.read_csv(files_dir + name, header=None, decimal='.', sep=',')
        df = df.T
        wn = df.to_numpy(dtype='float64')[0, 0:]
        df.drop(df.index[0], inplace=True)  # Retire les index et les nombres d'ondes
        df.insert(0, 'Classes', lab)  # Spécifie les labels pour chaques échantillons
        data = pd.concat([data, df], ignore_index=True)

    return data, wn
