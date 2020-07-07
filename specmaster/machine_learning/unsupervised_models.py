# mini-batch k-means clustering
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import time
import joblib


class SpectroKmeans:

    def __init__(self, n_cluster=2, rdm_ste=None, n_jobs=None):

        self.model = KMeans(n_clusters=n_cluster, random_state=rdm_ste, n_jobs=n_jobs)
        self.status = 'untrained'
        self.training_time = None

    def get_current_status(self):
        return self.status

    def get_training_duration(self):
        return self.training_time

    def fit_model(self, x_train):
        start_time = time.time()
        self.model.fit(x_train)
        self.training_time = time.time() - start_time
        self.status = 'trained'
        return self.model

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def scatter_plot(self, x_test, save_path=None):
        y_pred = self.model.predict(x_test)
        clusters = unique(y_pred)
        fig = plt.figure(figsize=(8, 4.5))
        with plt.style.context('default'):
            for cluster in clusters:
                # get row indexes for samples with this cluster
                row_ix = where(y_pred == cluster)
                # create scatter of these samples
                plt.scatter(x_test[row_ix, 0], x_test[row_ix, 1])
                plt.title('Kmeans Clustering')
        # show the plot
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300)

    def save_model(self, save_path):
        # save the model to disk
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)


class SpectroGmixture:

    def __init__(self, n_cluster=2, rdm_ste=None):
        self.model = GaussianMixture(n_components=n_cluster, random_state=rdm_ste)
        self.status = 'untrained'
        self.training_time = None

    def get_current_status(self):
        return self.status

    def get_training_duration(self):
        return self.training_time

    def fit_model(self, x_train):
        start_time = time.time()
        self.model.fit(x_train)
        self.training_time = time.time() - start_time
        self.status = 'trained'
        return self.model

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def scatter_plot(self, x_test, save_path=None):
        y_pred = self.model.predict(x_test)
        clusters = unique(y_pred)
        fig = plt.figure(figsize=(8, 4.5))
        with plt.style.context('default'):
            for cluster in clusters:
                # get row indexes for samples with this cluster
                row_ix = where(y_pred == cluster)
                # create scatter of these samples
                plt.scatter(x_test[row_ix, 0], x_test[row_ix, 1])
            plt.title('Gaussian Mixture Clustering')
        # show the plot
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300)

    def save_model(self, save_path):
        # save the model to disk
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
