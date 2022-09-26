"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides supervised learning models for vibrational spectra classification.
"""
from sklearn.ensemble import RandomForestClassifier as RandF
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


class _MachineLearningClassifier:
    """ Parent class for standard machine learning classifier.

    The models do not support binary labels; All binary labels are converted to integer labels.

    Parameters:
        model : scikit learn model
            Model defined in the child classes constructor (SpectroRF, SpectroSVM)
    """
    def __init__(self, model):
        # Uses the model defined in the child classes constructor (KMeans or GaussianMixture)
        self.model = model
        self.status = 'untrained'
        self.training_time = None

    def get_model(self):
        """ Returns the model."""
        return self.model

    def get_current_status(self):
        """ Returns the model current status """
        return self.status

    def get_training_duration(self):
        """ Returns the time required for the model training process."""
        return self.training_time

    def train_model(self, x_train, y_train):
        """ Train the model on a given set of spectra.

        Parameters:
            x_train : array
                Input Spectra. Array shape = (n_spectra, n_pixels).

            y_train : array
                Labels assigned the "x_train" spectra. Array shape = (n_spectra,) for integer labels and
                 (n_spectra, n_classes) for binary labels.
        """
        # Converts binary labels to integer labels. Does nothing if they are already integer labels.
        if y_train.ndim == 2 and y_train.shape[1] > 1:  # y_train is a binary matrix (one-hot encoded label)
            y_train = np.argmax(y_train, axis=1)

        start_time = time.time()
        self.model.fit(x_train, y_train)
        self.training_time = time.time() - start_time
        self.status = 'trained'

    def predict_classes(self, x_test):
        """ Predicts classes for the input spectra.

        Notes:
            This function must be preceded by the 'train_model()' function in order to properly work.

        Parameters:
            x_test : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and
                (n_pixels,) for a single spectrum.
        Return:
            (array) Predicted classes for input spectra. Array shape = (n_spectra,).
        """
        x_test = np.array(x_test, ndmin=2)  # x_test is forced to be a two-dimensional array
        y_pred = self.model.predict(x_test)  # returns the predicted classes
        return y_pred

    def predict_proba(self, x_test, averaged=False):
        """ Predicts the class probabilities for input spectra.

        Notes:
            This function must be preceded by the 'train_model()' function in order to properly work.

        Parameters:
            x_test : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and
                (n_pixels,) for a single spectrum.

            averaged : boolean, default=False
                If True, returns the mean and standard deviation of the predictions for each class.

        Return:
            (array) Predicted class probabilities for input spectra.
                 Array shape:
                    = (n_spectra, n_classes) if "averaged" = False.
                    = (1, n_classes) if "averaged" = True.
        """
        x_test = np.array(x_test, ndmin=2)  # x_test is forced to be a two-dimensional array
        y_pred = self.model.predict_proba(x_test)  # returns the predicted probabilities
        if averaged:
            y_pred = np.mean(y_pred, axis=0)
        return y_pred

    def get_classif_report(self, x_test, y_test, digits=4, class_names=None, save_path=None):
        """ Returns a classification report generated from a given set of spectra

        Notes:
            This function must be preceded by the 'train_model()' function in order to properly work.

        Parameters:
            x_test : array
                Input Spectra. Array shape = (n_spectra, n_pixels).

            y_test : array
                Labels assigned the "x_test" spectra. Array shape = (n_spectra,) for integer labels
                and(n_spectra, n_classes) for binary labels.

            digits : non-zero positive integer values, default=3
                Number of digits to display in the classification report.

            class_names : list or tupple of string, default=None
                Names or labels associated to the class. If None, class names are not displayed.

            save_path: string, default=None
                Path where the report is saved. If None, saving does not occur.

        Returns:
            Scikit Learn classification report
        """
        warnings.warn('\'get_classif_report\' instance method is no longer supported, use instead'
                      ' the standalone \'clf_report\' method found in boxsers.validation_metrics module.',
                      DeprecationWarning)
        # Converts binary labels to integer labels. Does nothing if they are already integer labels.
        if y_test.ndim == 2 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        # returns the predicted classes (format=integer label)
        y_pred = self.model.predict(x_test)
        # generates the classification report
        report = classification_report(y_test, y_pred, target_names=class_names, digits=digits)

        if save_path is not None:
            text_file = open(save_path, "w")
            text_file.write(report)
            text_file.close()
        return report

    def get_conf_matrix(self, x_test, y_test, normalize='true', class_names=None, title=None,
                        color_map='Blues', fmt='.2%', fontsize=10, fig_width=5.5, fig_height=5.5,
                        save_path=None):

        """ Returns a confusion matrix (built with scikit-learn) generated on a given set of spectra.

        Also produces a good quality image that can be saved and exported.

        Parameters:
            x_test : array
                Input Spectra. Array shape = (n_spectra, n_pixels).

            y_test : array
                Labels assigned the "x_test" spectra. Array shape = (n_spectra,) for integer labels
                and(n_spectra, n_classes) for binary labels.

            normalize : {'true', 'pred', None}, default=None
                - 'true': Normalizes confusion matrix by true labels(row)
                - 'predicted': Normalizes confusion matrix by predicted labels(col)
                - None: Confusion matrix is not normalized.

            class_names : list or tupple of string, default=None
                Names or labels associated to the class. If None, class names are not displayed.

            title : string, default=None
                Confusion matrix title. If None, there is no title displayed.

            color_map : string, default='Blues'
                Color map used for the confusion matrix heatmap.

            fmt: String, default='.2%'
                String formatting code for confusion matrix values. Examples:
                    - '.2f' = two floating values
                    - '.3%' = percentage with three floating values

            fontsize : positive float, default=10
                Font size(pts) used for the different of the confusion matrix.

            fig_width : positive float or int, default=5.5
                Figure width in inches.

            fig_height : positive float or int, default=5.5
                Figure height in inches.

            save_path : string, default=None
                Path where the figure is saved. If None, saving does not occur.

        Return:
            Scikit Learn confusion matrix
        """
        warnings.warn('\'get_conf_matrix\' instance method is no longer supported, use instead'
                      ' the standalone \'cf_matrix\' method found in boxsers.validation_metrics module.',
                      DeprecationWarning)
        # The model returns the predicted classes
        y_pred = self.model.predict(x_test)

        # Converts binary labels to integer labels. Does nothing if they are already integer labels.
        if y_test.ndim == 2 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        # scikit learn confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred, normalize=normalize)

        # creates a figure object
        fig = plt.figure(figsize=(fig_width, fig_height))
        # add an axes object
        ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
        # plot a Seaborn heatmap with the confusion matrix
        sns.heatmap(conf_matrix, annot=True, cmap=color_map, fmt=fmt, cbar=False, annot_kws={"fontsize": fontsize},
                    square=True)
        for _, spine in ax.spines.items():
            # adds a black outline to the confusion matrix
            spine.set_visible(True)
        # titles settings
        ax.set_title(title, fontsize=fontsize+1.2)  # sets the plot title, 1.2 points larger font size
        ax.set_xlabel('Predicted Label', fontsize=fontsize)  # sets the x-axis title
        ax.set_ylabel('True Label', fontsize=fontsize)  # sets the y-axis title
        if class_names is not None:
            # sets the xticks labels at an angle of 45°
            ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=fontsize)
            # sets the yticks labels vertically
            ax.set_yticklabels(class_names, rotation=0, fontsize=fontsize)
        # adjusts subplot params so that the subplot(s) fits in to the figure area
        fig.tight_layout()
        # save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()  # display the confusion matrix image
        return conf_matrix

    def save_model(self, save_path):
        """ Saves the model as (.sav) file to the given path."""
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        """ Load the model from the given path."""
        self.model = joblib.load(model_path)


class SpectroRF(_MachineLearningClassifier):
    """ Random forest classification model.

    Inherits several methods from the parent class "_MachineLearningClassifier".

    Parameters:
            n_trees : non-zero positive integer values, default=250
                Number of random trees to generate.

            rdm_ste : integer, default=None
                Random seed of the model. Using the same seed increases the chances of reproducing the same results.
    """
    def __init__(self, n_trees=250, rdm_ste=None):
        rf_model = RandF(n_estimators=n_trees, random_state=rdm_ste, n_jobs=2)
        # inherits the methods and arguments of the parent class _MachineLearningClassifier
        super().__init__(rf_model)


class SpectroSVM(_MachineLearningClassifier):
    """ Support Vector Machine classification model.

    Inherits several methods from the parent class "_MachineLearningClassifier".

    Parameters: TODO une autre fois
            rdm_ste : integer, default=None
                Random seed of the model. Using the same seed increases the chances of reproducing the same results.
    """
    def __init__(self, c=1.0, kernel='linear', gamma='scale', rdm_ste=None):
        svm_model = SVC(C=c, kernel=kernel, gamma=gamma, random_state=rdm_ste, probability=True)
        # inherits the methods and arguments of the parent class _MachineLearningClassifier
        super().__init__(svm_model)


class SpectroLDA(_MachineLearningClassifier):
    """ Linear Discriminant Analysis model object.

    Inherits several methods from the parent class "_MachineLearningClassifier".
    """
    def __init__(self):
        lda_model = LinearDiscriminantAnalysis()
        # inherits the methods and arguments of the parent class _MachineLearningClassifier
        super().__init__(lda_model)


if __name__ == "__main__":
    help(__name__)
