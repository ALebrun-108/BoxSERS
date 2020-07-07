from sklearn.ensemble import RandomForestClassifier as RandF
from sklearn.metrics import classification_report, confusion_matrix
from numpy import savetxt
import numpy as np
import joblib
import time


class SpectroRF:

    def __init__(self, n_trees=250, rdm_ste=None, n_jobs=-1):
        self.n_trees = n_trees
        self.rdm_ste = rdm_ste
        self.n_jobs = n_jobs
        self.model = RandF(n_estimators=self.n_trees, random_state=self.rdm_ste, n_jobs=self.n_jobs)
        self.status = 'untrained'
        self.training_time = None

    def get_current_status(self):
        return self.status

    def get_training_duration(self):
        return self.training_time

    def train_model(self, x_train, y_train):
        y_train = np.around(y_train)
        start_time = time.time()
        self.model.fit(x_train, y_train)
        self.training_time = time.time() - start_time
        self.status = 'trained'

    def predict_classes(self, x_test):
        x_test = np.array(x_test, ndmin=2)
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def predict_proba(self, x_test):
        # x_test is forced to be a two-dimensional array
        x_test = np.array(x_test, ndmin=2)
        # The model returns the probabilities
        y_preds = self.model.predict_proba(x_test)
        # Random forest returns a binary matrix (two columns) for each class.
        # The following opperation aims at combining the probabilities within a single matrix
        y_preds_array = np.zeros((x_test.shape[0], len(y_preds)))
        for i in range(len(y_preds)):
            y_preds_array[:, i] = y_preds[i][:, 1]
        return y_preds_array

    def get_classif_report(self, x_test, y_test, class_names=None, save_path=None):
        # Returns binary multiclass matrix
        y_pred = self.model.predict(x_test)
        # label binary --> integer conversion to perform classification report
        y_pred = np.argmax(y_pred, axis=1)

        report = classification_report(y_test, y_pred, target_names=class_names, digits=6)

        if save_path is not None:
            text_file = open(save_path, "w")
            text_file.write(report)
            text_file.close()
        return report

    def get_conf_matrix(self, x_test, y_test, normalize='true', class_names='',  save_path=None):
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        conf_matrix = confusion_matrix(y_test, y_pred, normalize=normalize)
        if save_path is not None:
            comment = 'Classes in order:'+str(class_names) + '\n Row = True classes, Col = Predicted classes'
            savetxt(save_path, conf_matrix,
                    delimiter=',',
                    fmt='%1.3f',
                    header=comment)
        return conf_matrix

    def get_feature_importances(self):
        return self.model.feature_importances_

    def save_model(self, save_path):
        # save the model to disk
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
