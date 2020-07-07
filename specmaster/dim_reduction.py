from sklearn.decomposition import PCA, FactorAnalysis, FastICA
import matplotlib.pyplot as plt
import joblib
import numpy as np

# TODO: Ajouter LCA et factor analysis et changer le nom pour data_reduction


class SpectroPCA:
    """
    Class for pca
    """
    def __init__(self, n_comp=10):
        self.n_comp = n_comp
        self.model = PCA(self.n_comp)

    def get_model(self):
        return self.model

    def fit_model(self, x_train):
        self.model.fit(x_train)
        return self.model

    def transform_spectra(self, x_test):
        x_pca = self.model.transform(x_test)
        return x_pca

    def explained_var_plot(self):
        expl_var = self.model.explained_variance_ratio_
        with plt.style.context('seaborn-notebook'):
            plt.plot(np.cumsum(expl_var) * 100)
            plt.xlabel('Number of PCA components')
            plt.ylabel('Cumulative explained variance (%)')
            plt.show()

    def scatter_plot(self, sp_test, lab, component_x=1, component_y=2, targets=None):
        """
        Scatter plot of the data as a function of the principal components(PC)

        Parameters:
            sp_test: Database with PC instead of initial pixel values.
            component_x (int): Principal component number used for the scatter plot x-axis.
            component_y (int): Principal component used for the scatter plot y-axis.
            lab: Labels of the spectra.
            targets: List containing the names of the classes to build the legend.

        Notes:
            Arguments component_x and component_y must be within the following range [1, self.n_comp]

        Returns:
            PCA scatter plot
        """

        if lab.ndim == 2:  # lab is a binary matrix (one-hot encoded label)
            lab = np.argmax(lab, axis=1)
            #lab = np.ones(shape=(sp_test.shape[0]))
        else:  # lab is a one-column vector with interger value
            lab = lab

        sp_test_red = self.model.transform(sp_test)
        unique = list(set(lab))
        c0 = component_x - 1
        c1 = component_y - 1
        plt.figure(figsize=(8, 4.5))
        with plt.style.context('default'):
            for i, u in enumerate(unique):
                xi = [sp_test_red[j, c0] for j in range(len(sp_test_red[:, c0])) if lab[j] == u]
                yi = [sp_test_red[j, c1] for j in range(len(sp_test_red[:, c1])) if lab[j] == u]
                plt.scatter(xi, yi, s=60, edgecolors='k', label=str(u))
            plt.xlabel('PC' + format(component_x))
            plt.ylabel('PC' + format(component_y))
            plt.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
            plt.title('Principal Component Analysis')
        plt.show()

    def component_plot(self, wn, component):
        """
        Principal component (PC) as a function of the different pixels of the spectra

        Parameters:
            wn: Wavenumber.
            component: Number of the PC under analysis.
        Example:
            component_plot(wn, component)

        Returns:
            Plot of a principal component vs wavelenght
        """
        pca_comp = self.model.components_
        pc_values = pca_comp[component - 1, :]
        with plt.style.context('seaborn-notebook'):
            fig, ax = plt.subplots()
            plt.plot(wn, pc_values.T)
            ax.set_xlabel('Wavenumber (cm' + r'$^{-1}$' + ')')
            ax.set_ylabel('PC' + format(component))
            ax.grid()
        plt.show()

    def save_model(self, save_path):
        # save the model to disk (.sav)
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)


class SpectroFA:
    """
    Class for pca
    """

    def __init__(self, n_comp=10):
        self.n_comp = n_comp
        self.model = FactorAnalysis(self.n_comp)

    def get_model(self):
        return self.model

    def fit_model(self, x_train):
        self.model.fit(x_train)
        return self.model

    def transform_spectra(self, x_test):
        x_fa = self.model.transform(x_test)
        return x_fa

    def scatter_plot(self, sp_test, lab, component_x=1, component_y=2, targets=None):
        """
        Scatter plot of the data as a function of the principal components(PC)

        Parameters:
            sp_test: Database with PC instead of initial pixel values.
            component_x (int): Principal component number used for the scatter plot x-axis.
            component_y (int): Principal component used for the scatter plot y-axis.
            lab: Labels of the spectra.
            targets: List containing the names of the classes to build the legend.

        Notes:
            Arguments component_x and component_y must be within the following range [1, self.n_comp]

        Returns:
            PCA scatter plot
        """

        if lab.ndim == 2:  # lab is a binary matrix (one-hot encoded label)
            lab = np.argmax(lab, axis=1)
            # lab = np.ones(shape=(sp_test.shape[0]))
        else:  # lab is a one-column vector with interger value
            lab = lab

        sp_test_red = self.model.transform(sp_test)
        unique = list(set(lab))
        c0 = component_x - 1
        c1 = component_y - 1
        plt.figure(figsize=(8, 4.5))
        with plt.style.context('default'):
            for i, u in enumerate(unique):
                xi = [sp_test_red[j, c0] for j in range(len(sp_test_red[:, c0])) if lab[j] == u]
                yi = [sp_test_red[j, c1] for j in range(len(sp_test_red[:, c1])) if lab[j] == u]
                plt.scatter(xi, yi, s=60, edgecolors='k', label=str(u))
            plt.xlabel('PC' + format(component_x))
            plt.ylabel('PC' + format(component_y))
            plt.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
            plt.title('Principal Component Analysis')
        plt.show()

    def component_plot(self, wn, component):
        """
        Principal component (PC) as a function of the different pixels of the spectra

        Parameters:
            wn: Wavenumber.
            component: Number of the PC under analysis.
        Example:
            component_plot(wn, component)

        Returns:
            Plot of a principal component vs wavelenght
        """
        pca_comp = self.model.components_
        pc_values = pca_comp[component - 1, :]
        with plt.style.context('seaborn-notebook'):
            fig, ax = plt.subplots()
            plt.plot(wn, pc_values.T)
            ax.set_xlabel('Wavenumber (cm' + r'$^{-1}$' + ')')
            ax.set_ylabel('PC' + format(component))
            ax.grid()
        plt.show()

    def save_model(self, save_path):
        # save the model to disk (.sav)
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)


class SpectroICA:
    """
    Class for pca
    """

    def __init__(self, n_comp=10):
        self.n_comp = n_comp
        self.model = FastICA(self.n_comp)

    def get_model(self):
        return self.model

    def fit_model(self, x_train):
        self.model.fit(x_train)
        return self.model

    def transform_spectra(self, x_test):
        x_fa = self.model.transform(x_test)
        return x_fa

    def scatter_plot(self, sp_test, lab, component_x=1, component_y=2, targets=None):
        """
        Scatter plot of the data as a function of the principal components(PC)

        Parameters:
            sp_test: Database with PC instead of initial pixel values.
            component_x (int): Principal component number used for the scatter plot x-axis.
            component_y (int): Principal component used for the scatter plot y-axis.
            lab: Labels of the spectra.
            targets: List containing the names of the classes to build the legend.

        Notes:
            Arguments component_x and component_y must be within the following range [1, self.n_comp]

        Returns:
            PCA scatter plot
        """

        if lab.ndim == 2:  # lab is a binary matrix (one-hot encoded label)
            lab = np.argmax(lab, axis=1)
            # lab = np.ones(shape=(sp_test.shape[0]))
        else:  # lab is a one-column vector with interger value
            lab = lab

        sp_test_red = self.model.transform(sp_test)
        unique = list(set(lab))
        c0 = component_x - 1
        c1 = component_y - 1
        plt.figure(figsize=(8, 4.5))
        with plt.style.context('default'):
            for i, u in enumerate(unique):
                xi = [sp_test_red[j, c0] for j in range(len(sp_test_red[:, c0])) if lab[j] == u]
                yi = [sp_test_red[j, c1] for j in range(len(sp_test_red[:, c1])) if lab[j] == u]
                plt.scatter(xi, yi, s=60, edgecolors='k', label=str(u))
            plt.xlabel('IC ' + format(component_x))
            plt.ylabel('IC ' + format(component_y))
            plt.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
            plt.title('Independent Component Analysis')
        plt.show()

    def component_plot(self, wn, component):
        """
        Principal component (PC) as a function of the different pixels of the spectra

        Parameters:
            wn: Wavenumber.
            component: Number of the PC under analysis.
        Example:
            component_plot(wn, component)

        Returns:
            Plot of a principal component vs wavelenght
        """
        pca_comp = self.model.components_
        pc_values = pca_comp[component - 1, :]
        with plt.style.context('seaborn-notebook'):
            fig, ax = plt.subplots()
            plt.plot(wn, pc_values.T)
            ax.set_xlabel('Wavenumber (cm' + r'$^{-1}$' + ')')
            ax.set_ylabel('PC' + format(component))
            ax.grid()
        plt.show()

    def save_model(self, save_path):
        # save the model to disk (.sav)
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)