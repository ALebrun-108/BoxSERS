"""
Author: Alexis Lebrun (Master's student)
School: Université Laval

The "ramanbox.visualization" module includes
"""
import matplotlib.pyplot as plt
import numpy as np


def random_plot(wn, sp, random_spectra, y_space=0, title=None, xlabel='Raman Shift (cm$^{-1}$)',
                ylabel='Intensity (a.u.)', line_width=1.5, line_style='solid', grid=True,
                darkstyle=False, fontsize=10, fig_width=6.08, fig_height=3.8, save_path=None):
    """
    Plot a number of randomly selected spectra from a set of spectra

    Parameters:
        wn : array
            X-axis(wavenumber, wavelenght, Raman shift, etc.) used for the
            spectra, array shape = (n_pixels, ).

        sp : array
            Input Spectra, array shape = (n_spectra, n_pixels).

        random_spectra : integer
            Number of spectrum that are randomly selected from "sp" to be plotted.

        y_space : int or float, default=0
            Extra space on the y-axis between the spectra to allow a better visualization of the spectra.

        title : string, default=None
            Font size(pts) used for the different elements of the graph. The title's font
            is two points larger than "fonctsize".

        xlabel : string, default='Raman Shift (cm$^{-1}$)'
            X-axis title. If None, there is no title displayed.

        ylabel : string, default='Intensity (a.u.)'
            Y-axis title. If None, there is no title displayed.

        line_width : positive float, default= 1.5
            Plot line width(s).

        line_style : string, default='solid'
            Plot line style(s).

        grid : boolean, default=True
            If True, a grid is displayed.

        darkstyle : boolean, default=False,
            If True, returns a plot with a dark background.

        fontsize : positive float, default=10
            Font size(pts) used for the different elements of the graph.

        fig_width : positive float or int, default=6.08
            Figure width in inches.

        fig_height : positive float or int, default=3.8
            Figure height in inches.

        save_path : string, default=None
            Path where the figure is saved. If None, saving does not occur.
    """
    if random_spectra > 0:
        if sp.ndim > 1:  # when this condition is not true, the parameter random_spectra is not taken into account
            if sp.shape[0] > random_spectra:
                random_row = np.random.choice(sp.shape[0], size=random_spectra, replace=False)
                sp = sp[random_row, :]
            else:
                raise ValueError('random_spectra size exceeded the number of samples in spectrum dataset')
        else:
            raise ValueError('Does not work and is useless with individual spectra')
    else:
        raise ValueError('random_spectra must be an integer greater than zero')

    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)
    y_space_sum = 0  # the y_space starts at zero
    for s in sp:
        # plots random spectra
        ax.plot(wn, s + y_space_sum, lw=line_width, ls=line_style)
        y_space_sum += y_space  # adds an offset for the next spectrum
    # titles settings
    ax.set_title(title, fontsize=fontsize+1.2)  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=fontsize)  # sets the x-axis label
    ax.set_ylabel(ylabel, fontsize=fontsize)  # sets the y-axis label
    # tick settings
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)  # 2.0 points smaller font size
    ax.tick_params(axis='both', which='minor')
    if grid is True:
        # adds a grid
        ax.grid(alpha=0.4)
    # adjusts subplot params so that the subplot(s) fits in to the figure area
    fig.tight_layout()
    # save figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def spectro_plot(wn, *sp, y_space=0, title=None, xlabel='Raman Shift (cm$^{-1}$)', ylabel='Intensity (a.u.)',
                 legend=None, line_width=1.5, line_style='solid', color=None, grid=True, darkstyle=False,
                 fontsize=10, fig_width=6.08, fig_height=3.8, save_path=None):
    """
    Returns a plot with the selected spectrum(s)

    Parameters:
        wn : array or list
            X-axis(wavenumber, wavelenght, Raman shift, etc.) used for the
            spectra, array shape = (n_pixels, ).

        *sp : array
            Input Spectrum(s), array shape = (n_spectra, n_pixels) for multiple spectra and (n_pixels,)
            for a single spectrum.

        y_space : int or float, default=0
            Extra space on the y-axis between the spectra to allow a better visualization of the spectra.

        title : string, default=None
            Font size(pts) used for the different elements of the graph. The title's font
             is two points larger than "fonctsize".

        xlabel : string, default='Raman Shift (cm$^{-1}$)'
            X-axis title. If None, there is no title displayed.

        ylabel : string, default='Intensity (a.u.)'
            Y-axis title. If None, there is no title displayed.

        legend : list or tupple of string, default=None
            Legend label(s). If None, the legend is not displayed.

        line_width : positive float, default= 1.5
            Plot line width(s).

        line_style : string, default='solid'
            Plot line style(s).

        color : string, default=None
            Plot line color(s).

        grid : boolean, default=False
            If True, a grid is displayed.

        darkstyle : boolean, default=False,
            If True, returns a plot with a dark background.

        fontsize : positive float, default=10
            Font size(pts) used for the different elements of the graph.

        fig_width : positive float or int, default=6.08
            Figure width in inches.

        fig_height : positive float or int, default=3.8
            Figure height in inches.

        save_path : string, default=None
            Path where the figure is saved. If None, saving does not occur.
    """

    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)
    y_space_sum = 0  # the y_space starts at zero

    if color is not None:
        ax.set_prop_cycle(color=color)
    for s in sp:
        # plots all the input spectra
        ax.plot(wn, s.T + y_space_sum, linewidth=line_width, linestyle=line_style, label='a')
        y_space_sum += y_space  # adds an offset for the next spectrum
    # titles settings
    ax.set_title(title, fontsize=fontsize+1.2)  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=fontsize)  # sets the x-axis title
    ax.set_ylabel(ylabel, fontsize=fontsize)  # sets the y-axis title
    # tick settings
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)  # 2.0 points smaller font size
    ax.tick_params(axis='both', which='minor')
    if grid is True:
        # adds a grid
        ax.grid(alpha=0.4)
    # adds a legend
    if legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(legend), fontsize=fontsize-2)
    # adjusts subplot params so that the subplot(s) fits in to the figure area
    fig.tight_layout()
    # save figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def class_plot(wn, sp, lab, y_space=0, std_color='grey', std_alpha=0.6,
               title=None, xlabel='Raman Shift (cm$^{-1}$)', ylabel='Intensity (a.u.)',
               class_names=None, line_width=1.5, line_style='solid', grid=True,
               fontsize=10, fig_width=6.08, fig_height=3.8,
               save_path=None):
    """
    Computes the mean and standard deviation of the spectra for each class and plots the averaged spectra
    with the areas covered by the standard deviations.

    Parameters:
        wn : array
            X-axis(wavenumber, wavelenght, Raman shift, etc.) used for the
            spectra, array shape = (n_pixels, ).

        sp : array
            Input Spectra, array shape = (n_spectra, n_pixels).

        lab : array
            Labels assigned the "sp" spectra, array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        y_space : int or float, default=0
            Extra space on the y-axis between the spectra to allow a better visualization of the spectra.

        std_color : string, default ='grey'
            Color of the standard deviation area

        std_alpha : float, default=0.6
            Transparency of the standard deviation area. Totally transparent if equal to 0
            and totally opaque if equal to 1.

        title : string, default=None
            Font size(pts) used for the different elements of the graph. The title's font
             is two points larger than "fonctsize".

        xlabel : string, default='Raman Shift (cm$^{-1}$)'
            X-axis title. If None, there is no title displayed.

        ylabel : string, default='Intensity (a.u.)'
            Y-axis title. If None, there is no title displayed.

        class_names : list or tupple of string, default=None
            Names or labels associated to the classes. If None, the legend is not displayed.

        line_width : positive float, default= 1.5
            Plot lines width.

        line_style : string, default='solid'
            Plot lines style.

        grid : boolean, default=True
            If True, a grid is displayed.

        fontsize : positive float, default=10
            Font size(pts) used for the different elements of the graph.

        fig_width : positive float or int, default=6.08
            Figure width in inches.

        fig_height : positive float or int, default=3.8
            Figure height in inches.

        save_path : string, default=None
            Path where the figure is saved. If None, saving does not occur.
    """
    # Converts binary labels to integer labels. Does nothing if they are already integer labels.
    if lab.ndim == 2 and lab.shape[1] > 1:  # lab is a binary matrix (one-hot encoded label)
        lab = np.argmax(lab, axis=1)

    unique = list(set(lab))
    if class_names is None:
        class_names = unique

    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)
    y_space_sum = 0  # the y_space starts at zero
    for i, u in enumerate(unique):
        # "i" does nothing
        # "u" corresponds to class labels
        # "j" give the position of each class
        sp_j = [sp[j, :] for j in range(len(sp[:, :])) if lab[j] == u]
        # mean and std calculation
        sp_mean = np.mean(sp_j, axis=0) + y_space_sum
        sp_std = np.std(sp_j, axis=0)
        # generates plot
        ax.plot(wn, sp_mean, lw=line_width, ls=line_style, label=class_names[u])
        # colors the area covered by the standard deviation.
        ax.fill_between(wn, sp_mean - sp_std, sp_mean + sp_std, facecolor=std_color, alpha=std_alpha)
        y_space_sum += y_space  # adds an offset for the next spectrum
    # titles settings
    ax.set_title(title, fontsize=fontsize+1.2)  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=fontsize)  # sets the x-axis title
    ax.set_ylabel(ylabel, fontsize=fontsize)  # sets the y-axis title
    # tick settings
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)  # 2.0 points smaller font size
    ax.tick_params(axis='both', which='minor')
    # adds a grid
    if grid:
        ax.grid(alpha=0.4)
    # adds a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), fontsize=fontsize-2)  # 2.0 points smaller font size
    # adjusts subplot params so that the subplot(s) fits in to the figure area
    fig.tight_layout()
    # save figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    help(__name__)
