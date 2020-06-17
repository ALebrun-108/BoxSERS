from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
import numpy as np


class SpectroPCA:
    """
    Class for pca
    """
    def __init__(self, sp_train, n_comp=10):
        self.sp_train = sp_train
        self.n_comp = n_comp
        self.pca_model = PCA(self.n_comp)
        self.pca_model.fit(self.sp_train)

    def get_model(self):
        return self.pca_model

    def transform_spectra(self, sp_test):
        sp_pca = self.pca_model.transform(sp_test)
        return sp_pca

    def explained_variance(self):
        expl_var = self.pca_model.explained_variance_ratio_
        with plt.style.context('seaborn-notebook'):
            plt.plot(np.cumsum(expl_var) * 100)
            plt.xlabel('Number of PCA components')
            plt.ylabel('Cumulative explained variance (%)')
            plt.show()

    def save_model(self, save_name):
        filename = save_name + '.sav'
        joblib.dump(self.pca_model, filename)

    def scatter_plot(self, sp_test, lab, comp_x=1, comp_y=2, targets=None):
        """
        Scatter plot of the data as a function of the principal components(PC)

        Parameters:
            sp_test: Database with PC instead of initial pixel values.
            comp_x (int): Principal component number used for the scatter plot x-axis.
            comp_y (int): Principal component used for the scatter plot y-axis.
            lab: Labels of the spectra.
            targets: List containing the names of the classes to build the legend.

        Notes:
            Arguments comp_x and comp_y must be within the following range [1, self.n_comp]

        Returns:
            PCA scatter plot
        """

        if lab.ndim == 2:  # lab is a binary matrix (one-hot encoded label)
            lab = np.argmax(lab, axis=1)
            #lab = np.ones(shape=(sp_test.shape[0]))
        else:  # lab is a one-column vector with interger value
            lab = lab

        sp_test_pca = self.pca_model.transform(sp_test)
        unique = list(set(lab))
        c0 = comp_x - 1
        c1 = comp_y - 1
        plt.figure(figsize=(8, 4.5))
        with plt.style.context('default'):
            for i, u in enumerate(unique):
                xi = [sp_test_pca[j, c0] for j in range(len(sp_test_pca[:, c0])) if lab[j] == u]
                yi = [sp_test_pca[j, c1] for j in range(len(sp_test_pca[:, c1])) if lab[j] == u]
                plt.scatter(xi, yi, s=60, edgecolors='k', label=str(u))
            plt.xlabel('PC' + format(comp_x))
            plt.ylabel('PC' + format(comp_y))
            plt.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
            plt.title('Principal Component Analysis')
        plt.show()

    def pca_component(self, wn, comp):
        """
        Principal component (PC) as a function of the different pixels of the spectra

        Parameters:
            wn: Wavenumber.
            comp: Number of the PC under analysis.
        Example:
            pca_component(wn, comp)

        Returns:
            Plot of a principal component vs wavelenght
        """
        pca_comp = self.pca_model.components_
        pc_values = pca_comp[comp - 1, :]
        with plt.style.context('seaborn-notebook'):
            fig, ax = plt.subplots()
            plt.plot(wn, pc_values.T)
            ax.set_xlabel('Wavenumber (cm' + r'$^{-1}$' + ')')
            ax.set_ylabel('PC' + format(comp))
            ax.grid()
        plt.show()
