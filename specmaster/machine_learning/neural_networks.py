import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution1D as Conv1D
from keras.layers import LocallyConnected1D as Lc1D
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, Activation
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import savetxt
import numpy as np
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SpectroCNN:

    def __init__(self):
        self.callbacks = []  # callbacks for the training
        self.status = 'untrained'
        self.model = Sequential()

    def reset_model(self):
        self.model = Sequential()
        self.status = 'untrained'

    def build_model(self, shape_in=1024, shape_out=2, architecture='Conv_Model', ks=5, output_activation='softmax',
                    dropout_rate=0.5):

        if architecture == 'Conv_Model':
            # conv block 1
            self.model.add(Conv1D(8, kernel_size=ks, input_shape=(shape_in, 1),
                                  padding='same', name='conv1'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp1'))
            # conv block 2
            self.model.add(Conv1D(16, ks, padding='same', name='conv2'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp2'))
            # conv block 3
            self.model.add(Conv1D(32, ks, padding='same', name='conv3'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp3'))
            # classification block
            self.model.add(Flatten())
            self.model.add(Dense(1024))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(1024))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(shape_out))
            self.model.add(Activation(output_activation))

        elif architecture == 'LC_Model':
            # conv block 1
            self.model.add(Conv1D(8, kernel_size=ks, input_shape=(shape_in, 1),
                                  padding='same', name='conv1'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp1'))
            # conv block 2
            self.model.add(Conv1D(16, ks, padding='same', name='conv2'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp2'))
            # conv block 3
            self.model.add(Lc1D(32, ks, padding='same', name='conv3'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp3'))
            # classification block
            self.model.add(Flatten())
            self.model.add(Dense(1024))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(1024))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(shape_out))
            self.model.add(Activation(output_activation))
        else:
            raise ValueError('Invalid architecture, load your own model or use: '
                             '{\'Conv_Model\',\'LC_Model\'}')

    def print_model_sumary(self):
        self.model.summary()

    def compile_model(self, optimizer='Adam', learning_rate=0.001, loss_function='Categorical'):
        """
        description

        Parameters:
             optimizer
             learning_rate
             loss_function

        Returns:

        """

        # optimizer
        if optimizer is 'Adam':
            opt = keras.optimizers.Adam(lr=learning_rate)
        elif optimizer is 'SGD':
            opt = keras.optimizers.SGD(lr=learning_rate)
        elif optimizer is 'SGD-Momentum':
            opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        else:
            raise NameError('Invalid optimize, valid choices: {\'Adam\', \'SGD\', \'SGD-Momentum\'}')

        # cost function
        if loss_function is 'Categorical':
            loss = keras.losses.categorical_crossentropy
        elif loss_function is 'Binary':
            loss = keras.losses.binary_crossentropy
        else:
            raise NameError('Invalid loss, valid choices: {\'Categorical\', \'Binary\'}')

        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    def config_earlystopping(self, monit='val_loss', epoch=5, verbose=1, best_weights=False):
        self.callbacks.append(EarlyStopping(monitor=monit,
                                            patience=epoch,
                                            verbose=verbose,
                                            restore_best_weights=best_weights))

    def config_modelcheckpoint(self, monit='val_loss', verbose=0,  best_only=True, save_path='best_model.h5'):
        self.callbacks.append(ModelCheckpoint(monitor=monit,
                                              filepath=save_path,
                                              verbose=verbose,
                                              save_best_only=best_only))

    def train_model(self, x_train, y_train, val_data=None, batch_size=64, ep=25, plot_history_enabled=True):
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        if val_data is not None:
            x_val = val_data[0]
            y_val = val_data[1]
            # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
            x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
            val_data = (x_val, y_val)

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=ep, verbose=1,
                                 callbacks=self.callbacks,
                                 # Validation data are passed for
                                 # monitoring validation loss and metrics
                                 # at the end of each epoch
                                 validation_data=val_data,
                                 shuffle=True)
        if plot_history_enabled:
            self.plot_history(history)

        self.status = 'trained'

    @staticmethod
    def plot_history(history):
        plt.subplots()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend(['Train Acc', 'Val Acc'], loc='best')
        plt.show()

        plt.subplots()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend(['Train loss', 'Val Loss'], loc='best')
        plt.show()

    def predict_classes(self, x_test):
        x_test = np.array(x_test, ndmin=2)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_pred = self.model.predict_classes(x_test)
        return y_pred

    def predict_proba(self, x_test):
        x_test = np.array(x_test, ndmin=2)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_pred = self.model.predict_proba(x_test)
        return y_pred

    def evaluate_loss(self, x_test, y_test):
        x_test = np.array(x_test, ndmin=2)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_pred = self.model.evaluate(x_test, y_test)
        return y_pred

    def get_classif_report(self, x_test, y_test, class_names=None, save_path=None):
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))  # Features modifications for CNN model
        y_pred = self.model.predict_classes(x_test)  # Returns integer values corresponding to the classes
        report = classification_report(y_test, y_pred, target_names=class_names, digits=6)
        if save_path is not None:
            text_file = open(save_path, "w")
            text_file.write(report)
            text_file.close()
        return report

    def get_conf_matrix(self, x_test, y_test, normalize='true', class_names='',  save_path=None):
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_pred = self.model.predict_classes(x_test)
        conf_matrix = confusion_matrix(y_test, y_pred, normalize=normalize)
        if save_path is not None:
            comment = 'Classes in order:'+str(class_names) + '\n Row = True classes, Col = Predicted classes'
            savetxt(save_path, conf_matrix,
                    delimiter=',',
                    fmt='%1.3f',
                    header=comment)
        return conf_matrix

    def features_extractor(self, x_test, layer_name):
        x_test = np.array(x_test, ndmin=2)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        layer = self.model.get_layer(name=layer_name)
        keras_function = keras.backend.function(self.model.input, layer.output)
        features = keras_function([x_test])
        return features

    def current_status(self):
        return self.status

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)


class SpectroDNN:

    def __init__(self, x_train, y_train, output_activation='softmax'):

        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        self.x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        self.y_train = y_train  # Training labels
        self.callbacks = []  # callbacks for the training
        self.output_activation = output_activation  # Output layer activation function
        self.status = 'untrained'

        self.model = Sequential(name='Dense Model')
        # conv block 1
        self.model.add(Dense(1024, input_shape=(np.shape(x_train)[1],), name='dense1'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(500))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.y_train.shape[1]))
        self.model.add(Activation(self.output_activation))

    def print_model_sumary(self):
        self.model.summary()

    def config_earlystopping(self, monit='val_loss', epoch=5, verbose=1, best_weights=False):
        self.callbacks.append(EarlyStopping(monitor=monit,
                                            patience=epoch,
                                            verbose=verbose,
                                            restore_best_weights=best_weights))

    def config_modelcheckpoint(self, monit='val_loss', verbose=0,  best_only=True, save_path='best_model.h5'):
        self.callbacks.append(ModelCheckpoint(monitor=monit,
                                              filepath=save_path,
                                              verbose=verbose,
                                              save_best_only=best_only))

    def compile_model(self, optimizer='Adam', learning_rate=0.001, loss='categorical_crossentropy'):
        if optimizer is 'Adam':
            opt = keras.optimizers.Adam(lr=learning_rate)
        elif optimizer is 'SGD':
            opt = keras.optimizers.SGD(lr=learning_rate)
        else:
            raise NameError('Invalid optimizer')

        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    def train_model(self, x_test, y_test, b_s=64, ep=25, plot_history_enabled=False):

        train_history = self.model.fit(self.x_train, self.y_train, batch_size=b_s, epochs=ep, verbose=1,
                                       shuffle=True, callbacks=self.callbacks, validation_data=(x_test, y_test))
        if plot_history_enabled:
            self.plot_history(train_history)
        self.status = 'trained'
        return train_history

    @staticmethod
    def plot_history(history):
        plt.subplots()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend(['Train Acc', 'Val Acc'], loc='best')
        plt.show()

        plt.subplots()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend(['Train loss', 'Val Loss'], loc='best')
        plt.show()

    def predict_classes(self, x_test):
        y_pred = self.model.predict_classes(x_test)
        return y_pred

    def predict_proba(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred

    def get_classif_report(self, x_test, y_test, class_names=None, save_path=None):
        cnn_scores = self.model.predict_classes(x_test)  # Returns integer values corresponding to the classes
        report = classification_report(y_test, cnn_scores, target_names=class_names, digits=6)

        if save_path is not None:
            text_file = open(save_path, "w")
            text_file.write(report)
            text_file.close()
        return report

    def get_conf_matrix(self, x_test, y_test, normalize='true', class_names='',  save_path=None):
        y_pred = self.model.predict_classes(x_test)
        conf_matrix = confusion_matrix(y_test, y_pred, normalize=normalize)
        if save_path is not None:
            comment = 'Classes in order:'+str(class_names) + '\n Row = True classes, Col = Predicted classes'
            savetxt(save_path, conf_matrix,
                    delimiter=',',
                    fmt='%1.3f',
                    header=comment)
        return conf_matrix

    def features_extractor(self, x_test, layer_name):
        x_test = np.array(x_test, ndmin=2)
        layer = self.model.get_layer(name=layer_name)
        keras_function = keras.backend.function(self.model.input, layer.output)
        features = keras_function([x_test])
        return features

    def current_status(self):
        return self.status

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)
