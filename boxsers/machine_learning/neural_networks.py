"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides neural network model specifically designed for the
classification of vibrational spectra.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
from skimage.transform import resize
import time
import matplotlib.pyplot as plt
import numpy as np
from boxsers._boxsers_utils import _lightdark_switch


class SpectroCNN:
    """ Convolutional Neural Network (CNN) for vibrational spectra classification.

    Parameters:
        shape_in : non-zero positive integer value
            Number of pixels in the spectra. The spectra must be uniform and have the same number of pixels.

        shape_out : non-zero positive integer value
            Number of output classes.

        ks : Odd positive integer value, default=5
            Size of kernel filters.

        dropout_rate : positive float integer between 0 and  1, default=0.4
            Dropout rate in dense layers.

        hidden_activation : string, default='relu'
            Hidden layer activation function.

        architecture : string, default='ConvModel'

        mode : {'multiclass', 'binary or multilabel'}, defaul='multiclass'
            Application mode for the model CNN that defines some of its hyperparameters. The loss function
            can be modified when compiling the model.
                - 'multiclass' : Output activation function = 'softmax', loss function = 'categorical'.
                - 'binary' or 'multilabel' :  Output activation function = 'sigmoid', loss function = 'binary'.
    """
    def __init__(self, shape_in, shape_out, ks=5, dropout_rate=0.3, hidden_activation='relu', architecture='ConvModel',
                 mode='multiclass'):
        # mode defintion
        if mode == 'multiclass':
            self.output_activation = 'softmax'
            self.loss_function = 'categorical_crossentropy'
        elif mode == 'binaryclass' or mode == 'multilabel':
            self.output_activation = 'sigmoid'
            self.loss_function = 'binary_crossentropy'
        else:
            raise ValueError('Invalid mode, valid choices: {\'multiclass\', \'multilabel\'}')

        # model architecture definition Todo: add sup. architecture
        if architecture == 'ConvModel':
            self.model = conv_model(shape_in, shape_out, nf_0=6, ks=ks, batchnorm=True, dropout_rate=dropout_rate,
                                    hidden_activation=hidden_activation, output_activation=self.output_activation)
        else:
            self.model = architecture

        self.learning_rate = 0.0001  # default learning rate =  1E-4
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)  # Default model optimizer = Adam
        self.metrics = ['acc']  # Default metrics = accuracy

        self.callbacks = []  # callbacks for the training (earlystopping and/or modelcheckpoint)
        self.history = None  # model last training history
        self.modelcheckpoint_path = None  # path where the model is saved if a modelcheckpoint is configured

        # python dictionary holding the principal hyperparameters of the model
        self.hyperparameter_dict = {"Input shape": shape_in, "Output shape": shape_out,
                                    "Kernel size": ks, "Batch Normalisation": True,
                                    "Drop out rate": dropout_rate,
                                    "Loss function": self.loss_function,
                                    "Hidden activation": hidden_activation,
                                    "Output activation": self.output_activation,
                                    "Optimizer": self.optimizer.__class__.__name__,
                                    "Learning rate": self.learning_rate,
                                    "Batch size": 0, "Epochs": 0,
                                    "Training duration": 0, "Status": 'Untrained',
                                    "Train sample": 0, "Validation sample": 0}
        # extraction of some hyperparameters from the child class

    def compile_model(self, optimizer=None, learning_rate=None, loss_function=None, metrics=None):
        """ Compile the CNN model with the latest changes made to the model.

        Notes:
            Parameters (other than None) submitted to this method replace the corresponding self.argument.

        Parameters:
            optimizer : keras.optimizers.Optimizer or str, default=None
                - Keras optimizer object : Replaces the current optimizer.
                - String : {'adam', 'sgd', 'sgd-momentum', None}
                    - 'adam' : Sets Adam algorithm(default parameters) as optimizer.
                    - 'sdg' : Sets Stochastic Gradient Descent(default parameters) as optimizer.
                    - 'sgd-momentum' : Sets Stochastic Gradient Descent(momentum=0.9, nesterov=True) as optimizer.
                - None : Keeps the current optimizer defined by "self.optimizer".

            learning_rate : non-zero positive float value, default=None
                Sets a new learning rate. If None, keeps the current learning rate defined by "self.learning_rate".

            loss_function : keras.losses.Losses or str, default=None
                - Keras loss function : Replaces the current loss function.
                - String : Keras loss function identifier or one of the following two options:
                    - 'categorical' : Sets categorical cross-entropy as loss function.
                    - 'binary' : Sets binary cross-entropy as loss function.
                - None : Keeps the current loss function defined by "self.loss_function".

            metrics : list of string, default=['acc']
                Keras metric returned during training and availlable in history
        """
        if metrics is not None:
            self.metrics = metrics

        # learning rate definition
        if learning_rate is not None:
            # a learning rate is given and replaces self.learning_rate
            self.learning_rate = learning_rate
            self.hyperparameter_dict["Learning rate"] = self.learning_rate

        # optimizer definition
        if optimizer is not None:
            # an optimizer is given and replaces the current self.optimizer
            if isinstance(optimizer, str):
                if optimizer.lower() == 'adam':
                    self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
                elif optimizer.lower() == 'sgd':
                    self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
                elif optimizer.lower() == 'sgd-momentum':
                    self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
            elif isinstance(optimizer, keras.optimizers.Optimizer):
                if learning_rate is not None:
                    # Override the learning rate of the provided optimizer
                    optimizer.learning_rate = learning_rate
                self.optimizer = optimizer
            else:
                raise ValueError('Invalid optimizer, must be a Keras optimizer object or a string value'
                                 ', valid choices: {\'adam\', \'sgd\', \'sgd-momentum\'}')

            self.hyperparameter_dict["Optimizer"] = self.optimizer.__class__.__name__

        # lost function definition
        if loss_function is not None:
            # a lost function is given and replaces self.lost_function
            if isinstance(loss_function, str):
                if loss_function.lower() == 'categorical':
                    self.loss_function = 'categorical_crossentropy'
                elif loss_function.lower() == 'binary':
                    self.loss_function = 'binary_crossentropy'
                else:
                    self.loss_function = loss_function
                self.hyperparameter_dict["Loss function"] = self.loss_function
            else:  # loss_function is not a string
                self.loss_function = loss_function
                self.hyperparameter_dict["Loss function"] = self.loss_function.__class__.__name__

        # model compilation
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)

    def print_info(self, structure=True, hyperparameters=True):
        """ Prints CNN model information.

        Parameters:
            structure : boolean, default=True
                If true, prints the structure of the CNN model with each of its layers.

            hyperparameters : boolean, default=True
                If true prints the hyperparameters of the CNN model
        """
        if structure is True:
            self.model.summary()
        if hyperparameters is True:
            for key, value in self.hyperparameter_dict.items():
                print(key, ' : ', value)
            print('\n')

    def get_model(self):
        """ Returns the model."""
        return self.model

    def config_earlystopping(self, epoch=10,  monit='val_loss'):
        """ Configures an early stopping during the CNN model training.

        Parameters:
            epoch : non-zero positive integer value, default=10
                Number of epochs to reach since the last CNN model improvement to stop the training.

            monit : {'acc', 'loss', 'val_acc', 'val_loss'}, default='val_loss'
                Metric used to determine whether or not the model is improving. The metrics 'val_acc'
                and 'val_loss' are not available if val_data=None in 'self.train_model' method.
        """
        es = EarlyStopping(monitor=monit, patience=epoch, verbose=1)
        self.callbacks.append(es)

    def config_modelcheckpoint(self, save_path='best_model', monit='val_acc'):
        """ Configures a checkpoint during CNN model training.

        Stores the model with the best performance and reinstores it at the end of training.

        Parameters:
            save_path : string, default='best_model
                Path where the model is saved

            monit : {'acc', 'loss', 'val_acc', 'val_loss'}, default='val_loss'
                Metric used to determine whether or not the model is improving. The metrics 'val_acc'
                and 'val_loss' are not available if val_data=None in 'self.train_model' method.
        """
        self.modelcheckpoint_path = save_path
        mc = ModelCheckpoint(monitor=monit, filepath=self.modelcheckpoint_path, verbose=2, save_best_only=True)
        self.callbacks.append(mc)

    def train_model(self, x_train, y_train, val_data=None, batch_size=92, n_epochs=25, reset_callbacks=True,
                    plot_history=True, verbose=1):
        """ Train the model on a given set of spectra.

        Parameters:
            x_train : array
                Spectra used to train the model. Array shape = (n_spectra, n_pixels).

            y_train : array
                Labels assigned to "x_train" spectra. Array shape = (n_spectra,) for integer labels and
                (n_spectra, n_classes) for binary labels.

            val_data : list or tuple of two numpy arrays, default=None
                - x_val : Spectra used to validate and monitor model performance during training.
                - y_val : Labels assigned the "x_train" spectra.
                Array shapes must follow the same format as x_train and y_train

            batch_size : non-zero positive integer value, default=92
                Batch size used for training update. Number of spectra used for each model
                modification during training.

            n_epochs : non-zero positive integer value, default=25
                Maximum number of model run throught the train set during training.

            reset_callbacks : boolean, default=True
                If True, reset the callback at the end of the training

            plot_history: boolean, default=True,
                If True, plot training history at the end of the training

            verbose: integer value, default=1
                Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

        Returns:
            Model training history.
        """
        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        # Also converts X-data np.array to tf.tensor
        x_train = tf.expand_dims(x_train, -1)

        validation = False
        n_val_samples = 0

        if val_data is not None:
            x_val = val_data[0]
            y_val = val_data[1]
            # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
            # Also converts X-data np.array to tf.tensor
            x_val = tf.expand_dims(x_val, -1)
            val_data = (x_val, y_val)
            n_val_samples = x_val.shape[0]
            validation = True

        start_time = time.time()
        self.history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=verbose,
                                      callbacks=self.callbacks, validation_data=val_data, class_weight=None,
                                      shuffle=True)

        training_time = time.time() - start_time

        if plot_history is True:
            # plot history at the end of the training.
            self.plot_history(self.metrics[0], validation=validation)

        if self.modelcheckpoint_path is not None:
            print('best model was uploaded')
            self.load_model(self.modelcheckpoint_path)

        if reset_callbacks is True:
            self.modelcheckpoint_path = None  # self.modelcheckpoint_path is reset
            self.callbacks = []  # callbacks are reset

        self.hyperparameter_dict["Status"] = 'Trained'
        self.hyperparameter_dict["Batch size"] = batch_size
        self.hyperparameter_dict["Epochs"] = n_epochs
        self.hyperparameter_dict["Training duration"] = training_time
        self.hyperparameter_dict["Train sample"] = x_train.shape[0]
        self.hyperparameter_dict["Validation sample"] = n_val_samples
        return self.history

    def plot_history(self, metric, validation=False, title='Training History', line_width=1.5, line_style='solid', darktheme=False,
                     grid=True, fontsize=10, fig_width=6.08, fig_height=5.7, save_path=None):
        """ Plot the history of the last model training for some performance metrics.

        Notes:
            This function must be preceded by the 'train_model()' function in order to properly work.

        Parameters:
            metric : string
                Metric to plot following model training. The metric needs to be defined when compiling
                the model with compile_model(..., metrics=).

            validation : boolean, default=False

            title : string, default='Training History'
                Plot title. If None, there is no title displayed.

            line_width : positive float, default=1.5
                Plot line width(s).

            line_style : string, default='solid' or '-'
                Plot line style(s).

            darktheme : boolean, default=False
                If True, returns a plot with a dark background

            grid : boolean, default=False
                If True, grids are displayed.

            fontsize : positive float, default=10
                Font size(pts) used for the different elements of the graph. The title's font
                is two points larger than "fonctsize".

            fig_width : positive float or int, default=6.08
                Figure width in inches.

            fig_height : positive float or int, default=5.7
                Figure height in inches.

            save_path : string, default=None
                Path where the figure is saved. If None, saving does not occur.
        """
        # update theme related parameters
        frame_color, bg_color, alpha_value = _lightdark_switch(darktheme)

        # creates a figure object and add two axes objects
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        fig.set_size_inches(fig_width, fig_height)

        ax1.plot(self.history.history[metric.lower()], linewidth=line_width, linestyle=line_style)
        ax2.plot(self.history.history['loss'], linewidth=line_width, linestyle=line_style)
        if validation:
            ax1.plot(self.history.history['val_'+metric.lower()], linewidth=line_width, linestyle=line_style)
            ax2.plot(self.history.history['val_loss'], linewidth=line_width, linestyle=line_style)

        ax1.set_title(title, fontsize=fontsize + 2, color=frame_color)
        ax1.set_ylabel(metric, fontsize=fontsize, color=frame_color)
        ax2.set_xlabel('Epoch #', fontsize=fontsize,  color=frame_color)
        ax2.set_ylabel('Loss', fontsize=fontsize, color=frame_color)

        # adds legends
        ax1.legend(['Train', 'Val'], loc='best', fontsize=fontsize,
                   facecolor=bg_color, labelcolor=frame_color)
        ax2.legend(['Train', 'Val'], loc='best', fontsize=fontsize,
                   facecolor=bg_color, labelcolor=frame_color)
        for ax in [ax1, ax2]:
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
            # adds grids
            if grid is True:
                ax.grid(alpha=alpha_value)

        # set figure and axes facecolor
        fig.set_facecolor(bg_color)
        ax1.set_facecolor(bg_color)
        ax2.set_facecolor(bg_color)
        # adjusts subplot params so that the subplot(s) fits in to the figure area
        fig.tight_layout()
        # save figure
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def predict_classes(self, x_test, return_integers=True, threshold=0.5, sums_classes=False):
        """ Predicts classes for the input spectra.

        Notes:
            This function must be preceded by the 'train_model()' function in order to properly work.

        Parameters:
            x_test : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra
                and (n_pixels,) for a single spectrum.

            return_integers : boolean, default=True
                If True, returns integer values instead of one-hot labels

            threshold : floating value between 0 and 1, default=0.5
                Classification threshold, gives 1 for predictions above it and 0 for predictions below
                it. Does nothing when return_integers is False.

            sums_classes : boolean, default=False
                If True, returns the total amount of detections for each class. If False, returns
                the detected classes for each input spectra. Does nothing when return_integers
                is True.

        Return:
            (array) Predicted classes for input spectra
                Array shape:
                    = (n_spectra,) if "return_integer" = True.
                    = (n_spectra, n_classes)  if "return_integer" = False and "sum_classes" = False.
                    = (1, n_classes) if "return_integer" = False and "sum_classes" = True.
        """
        # x_test initialization, x_test is forced to be a two-dimensional array
        x_test = np.array(x_test, ndmin=2)
        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        # Also converts X-data np.array to tf.tensor
        x_test = tf.expand_dims(x_test, -1)
        # x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        y_pred = self.model.predict(x_test)  # model application

        if return_integers:  # convert class labels to integer numbers
            return np.argmax(y_pred, axis=1)

        # converts the value above the threshold to 1 and the value below to 0
        y_pred = np.where(y_pred > threshold, 1, 0)

        if sums_classes:
            # sums the classes
            y_pred = np.sum(y_pred, axis=0)
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
            (array) Predicted class probabilities for input spectra
                 Array shape:
                    =(n_spectra, n_classes) if "averaged" = False.
                    =(1, n_classes) if "averaged" = True.
        """
        # x_test initialization, x_test is forced to be a two-dimensional array
        x_test = np.array(x_test, ndmin=2)
        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        # Also converts X-data np.array to tf.tensor
        x_test = tf.expand_dims(x_test, -1)
        y_pred = self.model.predict(x_test)

        if averaged:
            return np.mean(y_pred, axis=0)
        return y_pred

    def evaluate_acc_loss(self, x_test, y_test):
        """ Returns the loss and accuracy calculated on the given set of spectra

        Notes:
            This function must be preceded by the 'train_model()' function in order to properly work.

        Parameters:
            x_test : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and
                (n_pixels,) for a single spectrum.

            y_test : array
                Labels assigned to "x_test" spectra. Array shape = (n_spectra,) for integer labels
                and (n_spectra, n_classes) for binary labels.

        Returns:
            (float) Calculated loss value

            (float) Calculated accuracy value
        """
        # x_test initialization, x_test is forced to be a two-dimensional array
        x_test = np.array(x_test, ndmin=2)
        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        # Also converts X-data np.array to tf.tensor
        x_test = tf.expand_dims(x_test, -1)
        # loss evaluation
        loss, acc = self.model.evaluate(x_test, y_test)
        return loss, acc

    def evaluate_precision_recall_f1(self, x_test, y_test):
        """ Returns the precision, recall and F1 score calculated on the given set of spectra

        Notes:
            This function must be preceded by the 'train_model()' function in order to properly work.

        Parameters:
            x_test : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra and
                (n_pixels,) for a single spectrum.

            y_test : array
                Labels assigned to "x_test" spectra. Array shape = (n_spectra,) for integer labels
                and (n_spectra, n_classes) for binary labels.

        Returns:
            (float) Calculated precision value.

            (float) Calculated recall value.

            (float) Calculated F1 score value.
        """
        # x_test initialization, x_test is forced to be a two-dimensional array
        x_test = np.array(x_test, ndmin=2)
        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        # Also converts X-data np.array to tf.tensor
        x_test = tf.expand_dims(x_test, -1)

        # Converts binary labels to integer labels. Does nothing if they are already integer labels.
        if y_test.ndim == 2 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        # Returns integer values corresponding to the classes
        y_pred = self.predict_classes(x_test, return_integers=True)

        # Score calculations
        pre_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average='macro')[0:3]

        return pre_rec_f1

    def get_learned_features(self, x_test, layer_name='conv_2', resize_to_spectra=True):
        """
        Returns the feature maps captured by a model layer for a given input spectrum(s).

        todo: Give an overview in Github's ReadMe

        Parameters:
            x_test : array
                Input Spectrum(s). Array shape = (n_spectra, n_pixels) for multiple spectra
                and (n_pixels,) for a single spectrum.

            layer_name : string, default='conv_2'
                Label assigned to the layer selected.

            resize_to_spectra : boolean, default=True
                If True, resize the learned features to fit on the input spectrum(s).

        Returns:
            (array) Feature maps captured by a model layer for a given input spectrum(s).
                    Array hape = (n_spectra, n_pixels, n_filter) if resize_to_spectra=True
        """
        # x_test initialization, x_test is forced to be a two-dimensional array
        x_test = np.array(x_test, ndmin=2)
        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        # Also converts X-data np.array to tf.tensor
        x_test = tf.expand_dims(x_test, -1)

        layer = self.model.get_layer(name=layer_name)  # returns the chosen layer

        # access to intermediate output of the layer
        keras_function = keras.backend.function(self.model.input, layer.output)
        features = keras_function([x_test])  # shape=(n_spectrum, layer_input_size, n_filter)

        if resize_to_spectra:
            # Resize from (n_spectra, layer_input_size, n_filter) to (n_spectra, n_pixels, n_filter)
            features_resized = resize(features, [features.shape[0], x_test.shape[1], features.shape[2]], order=0)
        return features_resized

    def save_model(self, save_path):
        """ Saves the model with its architecture, the current weight values and the current optimizer. """
        self.model.save(save_path)

    def load_model(self, save_path):
        """ Loads the model. """
        self.model = load_model(save_path)

    @staticmethod
    def get_classif_report(self, x_test, y_test, digits=4, class_names=None, save_path=None):
        """
         No longer supported, use instead the standalone 'clf_report' method found in
         boxsers.validation_metrics module.
        """
        warnings.warn('\'get_classif_report\' instance method is no longer supported, use instead'
                      ' the standalone \'clf_report\' method found in boxsers.validation_metrics module.',
                      DeprecationWarning)

    @staticmethod
    def get_conf_matrix(x_test, y_test, normalize='true', class_names=None, title=None,
                        color_map='Blues', fmt='.2%', fontsize=10, fig_width=5.5, fig_height=5.5,
                        save_path=None):
        """
        No longer supported, use instead the standalone 'cf_matrix' method found in
        boxsers.validation_metrics module.
        """
        warnings.warn('\'get_conf_matrix\' instance method is no longer supported, use instead'
                      ' the standalone \'cf_matrix\' method found in boxsers.validation_metrics module.',
                      DeprecationWarning)


# Convolutional neural network architectures ----------------------------------------------------------
def conv_model(shape_in, shape_out, nf_0=6, dense_layers_size=None,  ks=5, batchnorm=True, dropout_rate=0.3,
               hidden_activation='relu', output_activation='softmax'):
    """
    Returns a CNN model with an architecture based on AlexNet.

        Fixed hyperparameters:
            - 3 conv layer, kernel filters number doubles from one convolutional layer to the next one.
            - 2 dense layer (1000, 500) neurons fixed

    Parameters:
        shape_in : non-zero positive integer value
            Number of pixels in the spectra. The spectra must be uniform and have the same number of pixels.

        shape_out : non-zero positive integer value
            Number of output classes.

        nf_0 : non-zero positive integer value, default=6
            Number of kernel filters applied in the first convolution.

        ks : Odd positive integer value, default=5
            Size of kernel filters.

        dense_layers_size : default=None

        batchnorm : boolean, default=True
            If True, apply batch normalization after Conv and Dense layers.

        dropout_rate : positive float integer between 0 and  1
            Dropout rate in dense layers.

        hidden_activation : string
            Hidden layer activation function.

        output_activation : string
            Output layer activation function

    Returns:
        Keras sequential model
    """
    if dense_layers_size is None:
        dense_layers_size = [1000, 500]

    inputs = keras.Input(shape=(shape_in, 1))
    x = inputs

    count = 0
    for filters in [nf_0, nf_0*2, nf_0*4]:
        x = layers.Conv1D(filters, ks, strides=1, padding="same", name='conv_'+str(count))(x)
        if batchnorm is True:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(hidden_activation)(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        count += 1
    x = layers.Flatten()(x)

    for units in dense_layers_size:
        x = layers.Dense(units)(x)
        if batchnorm is True:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(shape_out, activation=output_activation, name='output_layer')(x)
    model = keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    # help(__name__)
    cnn_model = conv_model(1024, 2, dense_layers_size=[1000, 1000])
    CNN = SpectroCNN(1024, 2, architecture=cnn_model, mode='multilabel')
    CNN.compile_model(optimizer='Adam', loss_function='binary', learning_rate=0.00005, metrics=['Precision'])
    CNN.print_info()
    a = CNN.train_model(np.random.random((100, 1024)), np.ones((100, 2)), n_epochs=50,
                        val_data=(np.random.random((100, 1024)), np.ones((100, 2))), verbose=1)

    for keys in a.history.keys():
        print(keys)
