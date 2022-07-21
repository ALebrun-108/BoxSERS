"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides different tools to visualize vibrational spectra quickly.
"""
import matplotlib.pyplot as plt
import numpy as np
from boxsers._boxsers_utils import _lightdark_switch


def random_plot(wn, sp, random_spectra, y_space=0, title=None, xlabel='Raman Shift (cm$^{-1}$)',
                ylabel='Intensity (a.u.)', line_width=1.5, line_style='solid', darktheme=False,
                grid=True, fontsize=10, fig_width=6.08, fig_height=3.8, save_path=None):
    """
    Plot a number of randomly selected spectra from a set of spectra.

    Parameters:
        wn : array
            X-axis(wavenumber, wavelenght, Raman shift, etc.), array shape = (n_pixels, ).

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

        darktheme : boolean, default=False
            If True, returns a plot with a dark background.

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
            Recommended format : .png, .pdf
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

    # update theme related parameters
    frame_color, bg_color, alpha_value = _lightdark_switch(darktheme)
    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
    y_space_sum = 0  # the y_space starts at zero
    for s in sp:
        # plots random spectra
        ax.plot(wn, s + y_space_sum, lw=line_width, ls=line_style)
        y_space_sum += y_space  # adds an offset for the next spectrum
    ax = plt.gca()  # get the current Axes instance

    # titles settings
    ax.set_title(title, fontsize=fontsize+1.2, color=frame_color)  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=fontsize, color=frame_color)  # sets the X-axis label
    ax.set_ylabel(ylabel, fontsize=fontsize, color=frame_color)  # sets the Y-axis label

    # tick settings
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major',
                   labelsize=fontsize-2,  # 2.0 points smaller font size
                   color=frame_color)
    ax.tick_params(axis='both', which='minor', color=frame_color)
    ax.tick_params(axis='x', colors=frame_color)  # setting up X-axis values color
    ax.tick_params(axis='y', colors=frame_color)  # setting up Y-axis values color
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color(frame_color)  # setting up spines color

    if grid is True:
        # adds a grid
        ax.grid(alpha=alpha_value)

    # set figure and axes facecolor
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    # adjusts subplot params so that the subplot(s) fits in to the figure area
    fig.tight_layout()
    # save figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def spectro_plot(wn, *sp, y_space=0, title=None, xlabel='Raman Shift (cm$^{-1}$)', ylabel='Intensity (a.u.)',
                 legend=None, line_width=1.5, line_style='solid', color=None, darktheme=False, grid=True,
                 fontsize=10, fig_width=6.08, fig_height=3.8, save_path=None):
    """
    Returns a plot with the selected spectrum(s)

    Parameters:
        wn : array or list
            X-axis(wavenumber, wavelenght, Raman shift, etc.), array shape = (n_pixels, ).

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

        darktheme : boolean, default=False
            If True, returns a plot with a dark background.

        grid : boolean, default=False
            If True, a grid is displayed.

        fontsize : positive float, default=10
            Font size(pts) used for the different elements of the graph.

        fig_width : positive float or int, default=6.08
            Figure width in inches.

        fig_height : positive float or int, default=3.8
            Figure height in inches.

        save_path : string, default=None
            Path where the figure is saved. If None, saving does not occur.
            Recommended format : .png, .pdf
    """
    # update theme related parameters
    frame_color, bg_color, alpha_value = _lightdark_switch(darktheme)

    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
    y_space_sum = 0  # the y_space starts at zero
    if color is not None:
        ax.set_prop_cycle(color=color)
    for s in sp:
        # plots all the input spectra
        ax.plot(wn, s.T + y_space_sum, lw=line_width, ls=line_style, label='a')
        y_space_sum += y_space  # adds an offset for the next spectrum
    ax = plt.gca()  # get the current Axes instance

    # titles settings
    ax.set_title(title, fontsize=fontsize + 1.2, color=frame_color)  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=fontsize, color=frame_color)  # sets the X-axis label
    ax.set_ylabel(ylabel, fontsize=fontsize, color=frame_color)  # sets the Y-axis label

    # tick settings
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major',
                   labelsize=fontsize - 2,  # 2.0 points smaller font size
                   color=frame_color)
    ax.tick_params(axis='both', which='minor', color=frame_color)
    ax.tick_params(axis='x', colors=frame_color)  # setting up X-axis values color
    ax.tick_params(axis='y', colors=frame_color)  # setting up Y-axis values color
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color(frame_color)  # setting up spines color

    if grid is True:
        # adds a grid
        ax.grid(alpha=alpha_value)

    # adds a legend
    if legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(legend), fontsize=fontsize-2,
                  facecolor=bg_color, labelcolor=frame_color)

    # set figure and axes facecolor
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
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
    (beta: work in progress)
    Computes the mean and standard deviation of the spectra for each class and plots the averaged spectra
    with the areas covered by the standard deviations.

    Parameters:
        wn : array
            X-axis(wavenumber, wavelenght, Raman shift, etc.), array shape = (n_pixels, ).

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
            Recommended format : .png, .pdf
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
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
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


def distribution_plot(lab, bar_width=0.8, avg_line=False, class_names=None,  title=None,  ylabel='Number of samples',
                      xlabel=None, color='orange', fontsize=10, fig_width=5.168, fig_height=3.23,
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
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
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


if __name__ == "__main__":
    help(__name__)
