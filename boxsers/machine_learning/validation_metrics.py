"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides different tools to evaluate the quality of a model’s predictions.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from boxsers._boxsers_utils import _lightdark_switch


def cf_matrix(y_true, y_pred, normalize='true', class_names=None, title=None,
              xlabel='Predicted label', ylabel='True label', darktheme=False,
              color_map='Blues', fmt='.2f', fontsize=10, fig_width=5.5, fig_height=5.5,
              save_path=None):
    """
    Returns a confusion matrix (built with scikit-learn) generated on a given set of spectra.
    Also produces a good quality image that can be saved and exported.

    Notes:
        When darktheme=True, it is recommended to use a colormap with a dark lower bound such as
        color_map='gray'.

    Parameters:
        y_true : array
            True labels. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        y_pred : array
            Predicted labels. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        normalize : {'true', 'pred', None}, default=None
            - 'true': Normalizes confusion matrix by true labels (row). Gives the recall scores
            - 'predicted': Normalizes confusion matrix by predicted labels (col). Gives the precision scores
            - None: Confusion matrix is not normalized.

        class_names : list or tupple of string, default=None
            Names or labels associated to the class. If None, class names are not displayed.

        title : string, default = None
            Confusion matrix title. If None, there is no title displayed.

        xlabel : string, default='Predicted label'
            X-axis title. If None, there is no title displayed.

        ylabel : string, default='True label'
            Y-axis title. If None, there is no title displayed.

        darktheme : boolean, default=False
            If True, returns a plot with a dark background.

        color_map : string, default = 'Blues'
            Color map used for the confusion matrix heatmap.

        fmt: String, default = '.2f'
            String formatting code for confusion matrix values. Examples:
                - '.0f' = integer
                - '.2f' = decimal with two floating values
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
    # Converts binary labels to integer labels. Does nothing if they are already integer labels.
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # scikit learn confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize)  # return a sklearn conf. matrix

    # update theme related parameters
    frame_color, bg_color, alpha_value = _lightdark_switch(darktheme)
    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
    # plot a Seaborn heatmap with the confusion matrix
    sns.heatmap(conf_matrix, annot=True, cmap=color_map, fmt=fmt, cbar=False, annot_kws={"fontsize": fontsize},
                square=True, )

    # titles settings
    ax.set_title(title, fontsize=fontsize + 1.2, color=frame_color)  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=fontsize, color=frame_color)  # sets the x-axis title
    ax.set_ylabel(ylabel, fontsize=fontsize, color=frame_color)  # sets the y-axis title

    # tick settings
    ax.tick_params(axis='both', which='major',
                   labelsize=fontsize - 2,  # 2.0 points smaller font size
                   color=frame_color)
    ax.tick_params(axis='x', colors=frame_color)  # setting up X-axis values color
    ax.tick_params(axis='y', colors=frame_color)  # setting up Y-axis values color

    for _, spine in ax.spines.items():
        # adds a black outline to the confusion matrix and setting up spines color
        spine.set_visible(True)
        spine.set_color(frame_color)

    if class_names is not None:
        # sets the xticks labels at an angle of 45°
        ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=fontsize-1.3,  # 1.2 points smaller font size
                           color=frame_color)
        # sets the yticks labels vertically
        ax.set_yticklabels(class_names, rotation=0, fontsize=fontsize-2, color=frame_color)

    # set figure and axes facecolor
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    # adjusts subplot params so that the subplot(s) fits in to the figure area
    fig.tight_layout()
    # save figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  # display the confusion matrix image
    return conf_matrix


def clf_report(y_true, y_pred, digits=4, print_report=True, class_names=None, save_path=None):
    """ Returns a classification report generated from a given set of spectra

    Notes:
        This function must be preceded by the 'train_model()' function in order to properly work.

    Parameters:
        y_true : array
            True labels. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        y_pred : array
            Predicted labels. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        digits : non-zero positive integer values, default=3
            Number of digits to display in the classification report.

        print_report : boolean, default=True
            If True, print the classification report

        class_names : list or tupple of string, default=None
            Names or labels associated to the class. If None, class names are not displayed.

        save_path: string, default=None
            Path where the report is saved. If None, saving does not occur.

    Returns:
        Scikit Learn classification report
    """
    # Converts binary labels to integer labels. Does nothing if they are already integer labels.
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # generates the classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=digits)

    if print_report:
        print(report)

    if save_path is not None:
        text_file = open(save_path, "w")
        text_file.write(report)
        text_file.close()
    return report


if __name__ == "__main__":
    help(__name__)
