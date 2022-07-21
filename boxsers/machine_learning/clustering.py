"""
Author : Alexis Lebrun (PhD. student)

School : Université Laval (Qc, Canada)

This module provides unsupervised learning models for vibrational spectra cluster analysis.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import joblib


class _UnsupervisedModel:
    """ Parent class for unsupervised learning model.

    Parameters:
        model : scikit learn model
            Model defined in the child classes constructor (SpectroKmmeans, SpectroGmixture)
    """
    def __init__(self, model):
        # Uses the model defined in the child classes constructor (KMeans or GaussianMixture)
        self.model = model
        self.status = 'untrained'  # Status = untrained or trained

    def get_current_status(self):
        """ Returns the model current status """
        return self.status

    def train_model(self, sp, y=None):
        """ Fits the model on a given set of spectra.

        Parameters:
            sp : array
                Input Spectra, array shape = (n_spectra, n_pixels).

            y : Ignored, default=None
                Not used, present here for API consistency by convention
        """
        self.model.fit(sp)
        self.status = 'trained'

    def predict_spectra(self, sp, y=None):
        """ Associates each spectrum with the closest model clusters

        Notes:
            This function must be preceded by the 'fit_model()' function in order to properly work.

        Parameters:
            sp : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and
                 (n_pixels,) for a single spectrum.

            y : Ignored, default=None
                Not used, present here for API consistency by convention
        Return:
            (array) Predicted clusters for each spectrum . Array shape = (n_spectra,)
        """
        sp = np.array(sp, ndmin=2)  # sp is forced to be a two-dimensional array
        y_pred = self.model.predict(sp)
        return y_pred

    def scatter_plot(self, sp, component_x=1, component_y=2, title=None, darkstyle=False, fontsize=10,
                     fig_width=6.08, fig_height=3.8, save_path=None):
        """ Returns a scatter plot of the spectra grouped in clusters.

        Notes:
            This function must be preceded by the "fit_model()" function in order to properly work.

        Parameters:
            sp : array
                Input Spectra. Array shape = (n_spectra, n_pixels).

            component_x : non-zero positive integer values, default=1
                Number of the component used for the scatter plot X-axis.

            component_y : non-zero positive integer values, default=2
                Number of the component used for the scatter plot Y-axis.

            title : string, default=None
                Plot title. If None, there is no title displayed.

            darkstyle : boolean, default=False,
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
        sp = np.array(sp, ndmin=2)  # sp is forced to be a two-dimensional array
        c0 = component_x - 1
        c1 = component_y - 1
        y_pred = self.model.predict(sp)
        clusters = np.unique(y_pred)

        # basic plot on a white background setting
        mpl_style = 'default'
        if darkstyle is True:
            # changes to a dark background plot
            mpl_style = 'dark_background'

        with plt.style.context(mpl_style):
            # creates a figure object
            fig = plt.figure(figsize=(fig_width, fig_height))
            # add an axes object
            ax = fig.add_subplot(1, 1, 1)
            for cluster in clusters:
                # get row indexes for samples with this cluster
                row_ix = np.where(y_pred == cluster)
                # create scatter of these samples
                ax.scatter(sp[row_ix, c0], sp[row_ix, c1], s=30, edgecolors='k', label=cluster)
            # title settings
            ax.set_title(title, fontsize=fontsize + 2)  # sets the plot title, 2 points larger font size
            ax.set_xlabel('Component ' + format(component_x), fontsize=fontsize)  # sets the x-axis title
            ax.set_ylabel('Component ' + format(component_y), fontsize=fontsize)  # sets the y-axis title
            # adds legend
            ax.legend(loc='best', fontsize=fontsize)
            # automatically adjusts subplot params so that the subplot(s) fits in to the figure area
            fig.tight_layout()
            # save figure
            if save_path is not None:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # show the plot
        plt.show()

    def save_model(self, save_path):
        """ Saves the model as (.sav) file to the given path."""
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        """ Load the model from the given path."""
        self.model = joblib.load(model_path)


class SpectroKmeans(_UnsupervisedModel):
    """ K-Means clustering model.

    Inherits several methods from the parent class "_UnsupervisedModel".

    Parameters:
            n_cluster : non-zero positive integer values, default=2
                Number of clusters to generate.

            rdm_ste : integer, default=None
                Random seed of the model. Using the same seed increases the chances of reproducing the same results.
    """
    def __init__(self, n_cluster=2, rdm_ste=None):
        kmeans_model = KMeans(n_clusters=n_cluster, init='k-means++', random_state=rdm_ste)
        # inherits the methods and arguments of the parent class _UnsupervisedModel
        super().__init__(kmeans_model)


class SpectroGmixture(_UnsupervisedModel):
    """ Gaussian mixture probability distribution model.

    Inherits several methods from the parent class "_UnsupervisedModel".

    Parameters:
            n_components : non-zero positive integer values, default=2
                Number of of mixture component to generate.

            rdm_ste : integer, default=None
                Random seed of the model. Using the same seed increases the chances of reproducing the same results.
    """
    def __init__(self, n_components=2, rdm_ste=None):
        gmixture_model = GaussianMixture(n_components=n_components, random_state=rdm_ste)
        # inherits the methods and arguments of the parent class _UnsupervisedModel
        super().__init__(gmixture_model)


if __name__ == "__main__":
    help(__name__)
