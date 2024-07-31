"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides different techniques to perform dimensionality reduction of
vibrational spectra.
"""
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pandas as pd
import seaborn as sns
from boxsers._boxsers_utils import _lightdark_switch


class _DimReductionModel:
    """ Parent class for vibrational spectra dimension reduction models.

    Parameters:
        model : scikit learn model
            Model defined in the child classes constructor (SpectroPCA, SpectroICA or SpectroFA)
    """
    def __init__(self, model):
        self.model = model

    def get_model(self):
        """ Returns the model."""
        return self.model

    def get_n_comp(self):
        """ Returns the number of components in the model. """
        return self.model.n_components

    def fit_model(self, sp):
        """ Fits the model on a given set of spectra.

        Parameters:
            sp : array
                Input Spectra. Array shape = (n_spectra, n_pixels).
        """
        self.model.fit(sp)  # fits the model on sp

    def transform_spectra(self, sp):
        """ Converts spectrum(s) into new components generated by the model.

        Notes:
            This function must be preceded by the 'fit_model()' function in order to properly work.

        Parameters:
            sp : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and
                (n_pixels,) for a single spectrum.
        Return:
            (array) Spectrum(s) converted to a reduced number of new components. Array shape = (n_spectra, n_pixels)
            for multiple spectra and (n_pixels,) for a single spectrum.
        """
        sp = np.array(sp, ndmin=2)  # sp is forced to be a two-dimensional array
        sp_redu = self.model.transform(sp)  # transforms and reduces the dimensions

        return sp_redu

    def pair_plot(self, sp, lab, n_components=3, title=None, save_path=None):
        """
        todo beta: work in progress
        Returns a seaborn pairplot to visualize the reduced dimmension

        Notes:
            - This function must be preceded by the "fit_model()" function in order to properly work.

        Parameters:
            sp : array
                Input Spectra. Array shape = (n_spectra, n_pixels).

            lab : array
                Labels assigned the "sp" spectra. Array shape = (n_spectra,) for integer labels and
                (n_spectra, n_classes) for binary labels.

            n_components: non-zero positive integer values, default=3
                Number of components used to make the pair plot.

            title : string, default=None
                Plot title. If None, there is no title displayed.

            save_path: string, default=None
                Path where the figure is saved. If None, saving does not occur.

        Returns:
            Scatter plot of the spectra as a function of two components of the model.
        """
        # Converts binary labels to integer labels. Does nothing if they are already integer labels.
        if lab.ndim == 2:  # lab is a binary matrix (one-hot encoded label)
            lab = np.argmax(lab, axis=1)

        component_labels = ['Component '+str(i) for i in range(1, n_components+1)]

        # transforms and reduces the dimensions
        sp_redu = pd.DataFrame(self.model.transform(sp)[:, 0:n_components], columns=component_labels)

        pca_df = pd.concat([sp_redu, pd.DataFrame({'Classes': lab})], axis=1)

        pair_plot = sns.pairplot(pca_df, palette='tab10', hue='Classes', aspect=1.6, plot_kws={'edgecolor': 'k'})
        # title settings (extra spacing of 1.1 inch is added)
        pair_plot.fig.suptitle(title, y=1.1)

        # save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def scatter_plot(self, sp, lab, component_x=1, component_y=2, class_names=None, title=None, marker_size=50,
                     palette='tab10', darktheme=False, grid=False, fontsize=10,
                     fig_width=5.06, fig_height=3.8, save_path=None):
        """ Returns a scatter plot of the spectra as a function of two new components produced by the model

        Notes:
            - This function must be preceded by the "fit_model()" function in order to properly work.
            - 'component_x' and 'component_y' must be less than or equal to the number of components in the model.

        Parameters:
            sp : array
                Input Spectra. Array shape = (n_spectra, n_pixels).

            lab : array
                Labels assigned the "sp" spectra. Array shape = (n_spectra,) for integer labels and
                (n_spectra, n_classes) for binary labels.

            component_x : non-zero positive integer values, default=1
                Number of the component used for the scatter plot X-axis.

            component_y : non-zero positive integer values, default=2
                Number of the component used for the scatter plot Y-axis.

            class_names : list or tupple of string, default=None
                Names or labels associated to the class. If None, the legend is not displayed.

            title : string, default=None
                Plot title. If None, there is no title displayed.

            marker_size : non-zero positive float values, default=50
                The marker size in points**2.

            palette : string, default = 'tab10'
                Palette used for the scatter plot.

            darktheme : boolean, default=False
                If True, returns a plot with a dark background.

            grid : boolean, default=False
                If True, a grid is displayed.

            fontsize : positive float, default=10
                Font size(pts) used for the different elements of the graph.

            fig_width : positive float or int, default=5.06
                Figure width in inches.

            fig_height : positive float or int, default=3.8
                Figure height in inches.

            save_path: string, default=None
                Path where the figure is saved. If None, saving does not occur.

        Returns:
            Scatter plot of the spectra as a function of two components of the model.
        """
        # Converts binary labels to integer labels. Does nothing if they are already integer labels.
        if lab.ndim == 2:  # lab is a binary matrix (one-hot encoded label)
            lab = np.argmax(lab, axis=1)

        if class_names is None:
            class_names = np.unique(lab)

        sp_redu = self.model.transform(sp)  # transforms and reduces the dimensions
        c0 = component_x - 1  # -1 since the indexes start at zero
        c1 = component_y - 1

        # update theme related parameters
        frame_color, bg_color, alpha_value = _lightdark_switch(darktheme)

        # creates a figure object
        fig = plt.figure(figsize=(fig_width, fig_height))
        # add an axes object
        ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
        sns.scatterplot(x=sp_redu[:, c0], y=sp_redu[:, c1], hue=lab, s=marker_size, palette=palette,
                        style=lab, edgecolor=frame_color)

        # title settings
        ax.set_title(title, fontsize=fontsize+1.2, color=frame_color)  # 1.2 points larger font size
        ax.set_xlabel('Component ' + format(component_x), fontsize=fontsize, color=frame_color)  # sets the x-axis title
        ax.set_ylabel('Component ' + format(component_y), fontsize=fontsize, color=frame_color)  # sets the y-axis title

        # tick settings
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major',
                       labelsize=fontsize - 2,  # 2.0 points smaller font size
                       color=frame_color)
        ax.tick_params(axis='both', which='minor', color=frame_color)
        ax.tick_params(axis='x', colors=frame_color)  # setting up X-axis values color
        ax.tick_params(axis='y', colors=frame_color)  # setting up Y-axis values
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color(frame_color)  # setting up spines color

        if grid is True:
            # adds a grid
            ax.grid(alpha=0.4)

        # legend settings
        handles, labels = ax.get_legend_handles_labels()  # get the legend handles
        ax.legend(handles, class_names,
                  fontsize=fontsize - 2,  # 2.0 points smaller font size
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

    def scatter_plot_3d(self, sp, lab, component_x=1, component_y=2, component_z=3, class_names=None, title=None,
                        darktheme=False, fontsize=10, fig_width=6.08, fig_height=3.8, save_path=None):
        """ Returns a scatter plot of the spectra as a function of two new components produced by the model

        Notes:
            - This function must be preceded by the "fit_model()" function in order to properly work.
            - 'component_x' and 'component_y' must be less than or equal to the number of components in the model.

        Parameters:
            sp : array
                Input Spectra. Array shape = (n_spectra, n_pixels).

            lab : array
                Labels assigned the "sp" spectra. Array shape = (n_spectra,) for integer labels and
                (n_spectra, n_classes) for binary labels.

            component_x : non-zero positive integer values, default=1
                Number of the component used for the scatter plot X-axis.

            component_y : non-zero positive integer values, default=2
                Number of the component used for the scatter plot Y-axis.

            component_z : non-zero positive integer values, default=3
                Number of the component used for the scatter plot Z-axis.

            class_names : list or tupple of string, default=None
                Names or labels associated to the class. If None, the legend is not displayed.

            title : string, default=None
                Plot title. If None, there is no title displayed.

            darktheme : boolean, default=False,
                If True, returns a plot with a dark background.

            fontsize : positive float, default=10
                Font size(pts) used for the different elements of the graph.

            fig_width : positive float or int, default=6.08
                Figure width in inches.

            fig_height : positive float or int, default=3.8
                Figure height in inches.

            save_path: string, default=None
                Path where the figure is saved. If None, saving does not occur.

        Returns:
            Scatter plot of the spectra as a function of two components of the model.
        """
        # todo: to be updated
        # Converts binary labels to integer labels. Does nothing if they are already integer labels.
        if lab.ndim == 2:  # lab is a binary matrix (one-hot encoded label)
            lab = np.argmax(lab, axis=1)

        sp_redu = self.model.transform(sp)  # transforms and reduces the dimensions

        unique = list(set(lab))
        c0 = component_x - 1
        c1 = component_y - 1
        c2 = component_z - 1

        # basic plot on a white background setting
        mpl_style = 'default'
        if darktheme is True:
            # changes to a dark background plot
            mpl_style = 'dark_background'

        with plt.style.context(mpl_style):
            # creates a figure object
            fig = plt.figure(figsize=(fig_width, fig_height))
            # add an axes object
            ax = fig.add_subplot(projection='3d')
            for i, u in enumerate(unique):
                # "i" does nothing
                # "u" corresponds to different class labels
                # "j" give the position of each class
                xi = [sp_redu[j, c0] for j in range(len(sp_redu[:, c0])) if lab[j] == u]
                yi = [sp_redu[j, c1] for j in range(len(sp_redu[:, c1])) if lab[j] == u]
                zi = [sp_redu[j, c2] for j in range(len(sp_redu[:, c2])) if lab[j] == u]
                ax.scatter(xi, yi, zi, s=30, edgecolors='k')
            # title settings
            ax.set_title(title, fontsize=fontsize + 1.2)  # sets the plot title, 1.2 points larger font size
            ax.set_xlabel('Component ' + format(component_x), fontsize=fontsize)  # sets the x-axis title
            ax.set_ylabel('Component ' + format(component_y), fontsize=fontsize)  # sets the y-axis title
            ax.set_zlabel('Component ' + format(component_z), fontsize=fontsize)  # sets the y-axis title
            # adds Classe Names in legend
            if class_names is not None:
                ax.legend(class_names, loc='best', fontsize=fontsize)
            # adjusts subplot params so that the subplot(s) fits in to the figure area
            fig.tight_layout()
            # save figure
            if save_path is not None:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def component_plot(self, wn, *component, title=None, xlabel='Raman Shift (cm$^{-1}$)',
                       ylabel='Component score (a.u.)', line_width=1.5, line_style='solid', darktheme=False,
                       color=None, grid=True, fontsize=10, fig_width=6.08, fig_height=3.8, save_path=None):
        """ Decomposition of a component as a function of spectral values

        Notes:
            This function must be preceded by the 'fit_model()' function in order to properly work.

        Parameters:
            wn : array
                X-axis(wavenumber, wavelenght, Raman shift, etc.) used for the spectra. Array shape = (n_pixels, ).

            *component : one or more non-zero positive integer values
                Index of the component(s) to plot.

            title : string, default=None
                Plot title. If None, there is no title displayed.

            xlabel : string, default='Raman Shift (cm$^{-1}$)'
                X-axis title. If None, there is no title displayed.

            ylabel : string, default='Component score (a.u.)'
                Y-axis title. If None, there is no title displayed.

            line_width : positive float, default= 1.5
                Plot line width(s).

            line_style : string, default='solid'
                Plot line style(s).

            color : string, default=None
                Plot line color(s).

            darktheme : boolean, default=False,
                If True, returns a plot with a dark background.

            grid : boolean, default=True
                If True, a grid is displayed.

            fontsize : positive float, default=10
                Font size(pts) used for the different elements of the graph.

            fig_width : positive float or int, default=6.08
                Figure width in inches.

            fig_height : positive float or int, default=3.8
                Figure height in inches.

            save_path: string, default=None
                Path where the figure is saved. If None, saving does not occur.

        Example:
            »» component_plot(raman_shift_array , 1, 3, 4)
                # Plots the scores of the first, third and fourth component for each raman shift.

        Returns:
            Plot of one or more components as a function of the x-axis of the spectra.
        """
        # update theme related parameters
        frame_color, bg_color, alpha_value = _lightdark_switch(darktheme)
        # creates a figure object
        fig = plt.figure(figsize=(fig_width, fig_height))
        # add an axes object
        ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
        if color is not None:
            ax.set_prop_cycle(color=color)
        # gets the components of the model decomposed according to the x-values of the spectra
        model_comp = self.model.components_
        for c in component:
            # plots all the input components
            model_scores = model_comp[c - 1, :]
            ax.plot(wn, model_scores.T, label='Component ' + format(c), lw=line_width, ls=line_style)

        # title settings
        ax.set_title(title, fontsize=fontsize+1.2, color=frame_color)  # 1.2 points larger font size
        ax.set_xlabel(xlabel, fontsize=fontsize, color=frame_color)  # sets the x-axis label
        ax.set_ylabel(ylabel, fontsize=fontsize, color=frame_color)  # sets the y-axis label

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
        # adds a grid
        if grid:
            ax.grid(alpha=alpha_value)
        # adds a legend
        ax.legend(loc='best', fontsize=fontsize-2,  # 2.0 points smaller font size
                  facecolor=bg_color, labelcolor=frame_color)
        # set figure and axes facecolor
        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        # automatically adjusts subplot params so that the subplot(s) fits in to the figure area
        fig.tight_layout()
        # save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, save_path):
        """ Saves the model as (.sav) file to the given path."""
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        """ Load the model from the given path."""
        self.model = joblib.load(model_path)


""" Child Classes --------------------------------------------"""


class SpectroPCA(_DimReductionModel):
    """ Principal Component Analysis(PCA) model object.

    Uses the PCA class provided by scikit-learn, but it's optimize for spectral analysis and
    adds new functionality, including visual features to improve the analysis.

    Inherits several methods from the parent class "_DimReductionModel".

    Parameters:
            n_comp : non-zero positive integer values, default=10
                Number of model components to use.
    """
    def __init__(self, n_comp=10):
        pca_model = PCA(n_comp)  # sklearn PCA model
        # inherits the methods and arguments of the parent class _DimReductionModel
        super().__init__(pca_model)

    def explained_var_plot(self, title=None, xlabel='Number of PCA components',
                           ylabel='Cumulative explained variance (%)', color=None, line_width=1.5,
                           line_style='solid', darktheme=False, grid=True, fontsize=10,
                           fig_width=6.08, fig_height=3.8, save_path=None):
        """ Plot the cumulative explained variance(%) as a function of the number of principal components(PC)

        Notes:
            This function must be preceded by the 'fit_model()' function in order to properly work.

        Parameters:
            title : string, default=None
                Plot title. If None, there is no title displayed.

            xlabel : string, default='Raman Shift (cm$^{-1}$)'
                X-axis title. If None, there is no title displayed.

            ylabel : string, default='Component score (a.u.)'
                Y-axis title. If None, there is no title displayed.

            color : string, default=None
                Plot line color(s).

            line_width : positive float, default= 1.5
                Plot line width(s).

            line_style : string, default='solid'
                Plot line style(s).

            darktheme : boolean, default=False,
                If True, returns a plot with a dark background.

            grid : boolean, default=True
                If True, a grid is displayed.

            fontsize : positive float, default=10
                Font size(pts) used for the different elements of the graph.

            fig_width : positive float or int, default=6.08
                Figure width in inches.

            fig_height : positive float or int, default=3.8
                Figure height in inches.

            save_path: string, default=None
                Path where the figure is saved. If None, saving does not occur.
        """
        expl_var = self.model.explained_variance_ratio_

        # update theme related parameters
        frame_color, bg_color, alpha_value = _lightdark_switch(darktheme)
        # creates a figure object
        fig = plt.figure(figsize=(fig_width, fig_height))
        # add an axes object
        ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
        # plot the graph
        ax.plot(np.cumsum(expl_var) * 100, color=color, lw=line_width, ls=line_style)

        # title settings
        ax.set_title(title, fontsize=fontsize + 1.2, color=frame_color)  # 1.2 points larger font size
        ax.set_xlabel(xlabel, fontsize=fontsize, color=frame_color)  # sets the x-axis label
        ax.set_ylabel(ylabel, fontsize=fontsize, color=frame_color)  # sets the y-axis label

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
        # adds a grid
        if grid:
            ax.grid(alpha=alpha_value)

        # set figure and axes facecolor
        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        # automatically adjusts subplot params so that the subplot(s) fits in to the figure area
        fig.tight_layout()
        # save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class SpectroICA(_DimReductionModel):
    """ Independant Component Analysis(ICA) model object.

    Uses the FastICA class provided by scikit-learn, but it's optimize for spectral analysis and
    adds new functionality, including visual features to improve the analysis.

    Inherits several methods from the parent class "_DimReductionModel".

    Notes:
        - FastICA is a fast algorithm for Independent Component Analysis whose implementation is based on
          https://doi.org/10.1016/S0893-6080(00)00026-5

    Parameters:
            n_comp : non-zero positive integer values, default=10
                Number of model components to use.
    """
    def __init__(self, n_comp=10):
        ica_model = FastICA(n_comp)  # sklearn FastICA model
        # inherits the methods and arguments of the parent class _DimReductionModel
        super().__init__(ica_model)


if __name__ == "__main__":
    help(__name__)
