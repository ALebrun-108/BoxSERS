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


class CnnBeta:

    def __init__(self, x_train, y_train, ks=5, output_activation='softmax', architecture='Conv_Model'):

        self.ks = ks  # Kernel filter size
        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        self.x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        self.y_train = y_train  # Training labels
        self.callbacks = []  # callbacks for the training
        self.output_activation = output_activation  # Output layer activation function

        if architecture == 'Conv_Model':
            self.model = Sequential(name='Conv_Model')
            # conv block 1
            self.model.add(Conv1D(8, kernel_size=self.ks, input_shape=(np.shape(x_train)[1], 1),
                                  padding='same', name='conv1'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp1'))
            # conv block 2
            self.model.add(Conv1D(16, self.ks, padding='same', name='conv2'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp2'))
            # conv block 3
            self.model.add(Conv1D(32, self.ks, padding='same', name='conv3'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp3'))
            # classification block
            self.model.add(Flatten())
            self.model.add(Dense(1024))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1024))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.y_train.shape[1]))
            self.model.add(Activation(self.output_activation))

        elif architecture == 'LC_Model':
            self.model = Sequential(name='LC_Model')
            # conv block 1
            self.model.add(Conv1D(8, kernel_size=self.ks, input_shape=(np.shape(x_train)[1], 1),
                                  padding='same', name='conv1'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp1'))
            # conv block 2
            self.model.add(Conv1D(16, self.ks, padding='same', name='conv2'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp2'))
            # conv block 3
            self.model.add(Lc1D(16, self.ks, name='conv3'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=2, name='maxp3'))
            # classification block
            self.model.add(Flatten())
            self.model.add(Dense(1024))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dense(1024))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.y_train.shape[1]))
            self.model.add(Activation(self.output_activation))
        else:
            raise ValueError('Invalid architecture, load your own model or use: '
                             '{\'Conv_Model\',\'LC_Model\'}')

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

    def train_model(self, x_test, y_test, b_s=64, ep=25, plot_history_enabled=True):

        # Features modifications for CNN model: shape_initial = (a,b) --> shape_final = (a,b,1)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        self.history = self.model.fit(self.x_train, self.y_train,
                                      batch_size=b_s,
                                      epochs=ep,
                                      verbose=1,
                                      shuffle=True,
                                      callbacks=self.callbacks,
                                      validation_data=(x_test, y_test))

        if plot_history_enabled:
            self.plot_history()


    def plot_history(self,):
        plt.subplots()
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend(['Train Acc', 'Val Acc'], loc='best')
        plt.show()

        plt.subplots()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend(['Train loss', 'Val Loss'], loc='best')
        plt.show()

    def predict_classes(self, x_test):
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_pred = self.model.predict_classes(x_test)
        return y_pred

    def predict_proba(self, x_test):
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_pred = self.model.predict_proba(x_test)
        return y_pred

    def get_classif_report(self, x_test, y_test, class_names=None, save_path=None):
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))  # Features modifications for CNN model
        cnn_scores = self.model.predict_classes(x_test)  # Returns integer values corresponding to the classes
        report = classification_report(y_test, cnn_scores, target_names=class_names, digits=6)

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

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)


if __name__ == '__main__':
    callbacks = []
    a = EarlyStopping(patience=2)
    b = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)
    callbacks.append(a)
    callbacks.append(b)

    callbacks2 = []
    a = EarlyStopping(patience=2)
    b = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)
    callbacks2.append(EarlyStopping(patience=2))
    callbacks2.append(ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True))
