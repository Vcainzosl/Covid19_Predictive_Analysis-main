import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from urllib.request import urlretrieve
from window_slider import Slider

from utils.WindowData import WindowData
from utils.Saving import Saving


class Preprocessing(Saving):
    """Manage methods for cleaning, customizing and adapting dataset according to requirements"""

    def __init__(
        self,
        base_dir=".",
        csv=True,
        img=True,
        pickle=False,
        input_folder="Datasets",
    ):
        super().__init__(base_dir=base_dir, csv=csv, img=img, pickle=pickle)
        self.input_folder = input_folder
        self.input_dir = self.base_dir + os.path.sep + self.input_folder

    def download_dataset(self, url: str, **kwargs) -> str:
        """Performs downloading datasets from url sources, creating and saving in respective directory

        :param url: imformation source domain name server
        :type url: str
        :return: dataset filepath
        :rtype: str
        """
        if not url:
            return
        os.makedirs(self.input_dir, **kwargs)
        filename = url.split("/")[-1]
        filepath = (
            self.input_dir
            + os.path.sep
            + Saving.sort_file(filename, self.input_dir)
            + filename
        )
        urlretrieve(url, filepath)
        return filepath

    def read_dataset(
        self,
        filepath: str,
        filename="Original Data",
        **kwargs,
    ) -> pd.DataFrame:
        """Reads dataset from .csv file

        :param filepath: dataset path
        :type filepath: str
        :param header: insert DataFrame header 1 does 0 not, defaults to 0
        :type header: int, optional
        :param index_col: insert index column 1 does 0 not, defaults to 0
        :type index_col: int, optional
        :param filename: name of the original DataSet .csv file
        :type filename: str
        :return: dataset rendered to DataFrame
        :rtype: pd.DataFrame
        """
        data = pd.read_csv(filepath, **kwargs)
        self.save_csv(data.head().to_csv(), filename)
        return data

    def filter_by_names(
        self,
        data: pd.DataFrame,
        column: str,
        values: list,
        filename="Filter Data",
        **kwargs,
    ) -> pd.DataFrame:
        """Filter dataset samples by names in columns

        :param data: DataSet with samples
        :type data: pd.DataFrame
        :param column: column of the DataSet within values to filter
        :type column: str
        :param values: elements in column, list of values data type
        :type values: list
        :param filename: name of the DataSet filtered .csv file
        :type filename: str
        :return: DataSet filtered
        :rtype: pd.DataFrame
        """
        data = data[data[column].isin(values)]
        data = data.groupby(data.index.name, **kwargs)[data.columns].sum()
        self.save_csv(data.head().to_csv(), filename)
        return data

    def eliminate_variables(
        self,
        data: pd.DataFrame,
        columns: list,
        filename="Elimination Data",
        **kwargs,
    ) -> pd.DataFrame:
        """delete those columns in DataSet unwanted

        :param data: Dataset with samples
        :type data: pd.DataFrame
        :param column: names of variables to delete
        :type column: list
        :param filename: name of the DataSet after elimination .csv file
        :type filename: str
        :return: Dataset clean of undesired features
        :rtype: pd.DataFrame
        """
        data = data.drop(columns, **kwargs)
        self.save_csv(data.head().to_csv(), filename)
        return data

    def window_slide_dataset(
        self,
        data: pd.DataFrame,
        bucket_size: int,
        overlap_count: int,
        wrapper_class=WindowData,
        filename="Slided Data",
        **kwargs,
    ) -> pd.DataFrame:
        """runs sliding window with overlapping on DataSet values
        :param data: dataset with sample
        :type data: pd.DataFrame
        :param bucket_size: window size, number of samples taking into the window
        :type bucket_size: int
        :param overlap_count: number of previous window elements considered in the next step, must be lesser than bucket_size
        :type overlap_count: int
        :param filename: name of the DataSet slided .csv file
        :type filename: str
        :return: Forecasting DataFrame from time series
        :rtype: pd.DataFrame
        """

        feature_values = data.T.values
        bucket_size = bucket_size
        overlap_count = overlap_count

        samples = []
        columns = []
        for name in data.columns:
            for i in range(overlap_count):
                columns.append(name + "_t-" + str(overlap_count - i))
            columns.append(name)

        slider = Slider(bucket_size, overlap_count, wrapper_class)
        slider.fit(feature_values)
        while True:
            window_data = slider.slide()
            if slider.reached_end_of_list():
                break
            samples.append(window_data.build_sample())
        data = pd.DataFrame(
            samples,
            columns=columns,
            index=data.T.columns[bucket_size - 1 :],
            **kwargs,
        )
        self.save_csv(
            data.head().to_csv(),
            filename + "-windowsize=" + str(bucket_size),
        )
        return data

    @staticmethod
    def split_data(data: pd.DataFrame, t_label: str, axis=1, **kwargs):
        t = data[t_label]
        X = data.drop(t_label, axis=axis, **kwargs)

        return X, t

    def get_plots(self, data: pd.DataFrame):
        fig = plt.figure(len(data.columns), figsize=(10, 5))
        ind = pd.to_datetime(data.index.tolist())
        plt.plot(ind[:], data.iloc[:, 0].values)
        plt.title(data.columns[0])
        plt.legend([data.columns[0]], loc="upper right")
        plt.tight_layout()
        self.save_img(plt, "Distribution Data")

        for i in range(len(data.columns) - 1):
            plt.clf()
            plt.plot(ind[:], data.iloc[:, i + 1].values)
            plt.title(data.columns[i + 1])
            plt.legend([data.columns[i + 1]], loc="upper right")
            plt.tight_layout()
            self.save_img(plt, "Distribution Data(" + str(i + 1) + ")")
        plt.close()

    def get_correlation_matrix(self, X):
        samples, nvar = X.shape
        plt.figure(figsize=(7, 6))
        # Matriz de correlación usando las primeras 30 variables
        corr_mat = np.corrcoef(np.c_[X].T)
        etiquetas = X.columns.values.tolist()
        # Mapa de calor sobre las variables de la matriz de correlación
        sns.heatmap(
            corr_mat,
            vmin=-1,
            vmax=1,
            annot=True,
            linewidths=1,
            cmap="BrBG",
            xticklabels=etiquetas,
            yticklabels=etiquetas,
        )
        plt.tight_layout()
        self.save_img(plt, "Correlation matrix")
        plt.close()

    def get_PCA(self, X, **kwargs):
        samples, nvar = X.shape
        pca = PCA(**kwargs)
        pca.fit(X)
        plt.figure(figsize=(10, 5))
        plt.bar(
            X.columns.values.tolist(),
            pca.explained_variance_ratio_ * 100,
            color="b",
            align="center",
            tick_label=X.columns.values.tolist(),
        )
        plt.xticks(rotation="vertical")
        indices = np.argsort(pca.explained_variance_ratio_)
        plt.xlabel("Componentes principales")
        plt.ylabel("% de varianza explicada")
        plt.tight_layout()
        self.save_img(plt, "PCA analisys")
        plt.close()

    def get_ICA(self, X, **kwargs):
        ica = FastICA(**kwargs)
        comp = ica.fit_transform(X)
        f, c = X.shape
        ind = pd.to_datetime(X.index.tolist())
        fig = plt.figure(figsize=(10, 5))
        ind = pd.to_datetime(X.index.tolist())
        plt.plot(ind[:], X.values)
        plt.title("Conjunto de variables")
        plt.legend(X.columns, loc="upper right")

        plt.tight_layout()

        self.save_img(plt, "ICA Analisys")

        fig = plt.figure(figsize=(10, 5))
        ind = pd.to_datetime(X.index.tolist())
        for i in range(len(comp[0])):
            plt.plot(ind[:], comp[:, i], label="IC" + str(i + 1))
        plt.title("Componentes independientes")
        plt.legend(loc="upper right")
        plt.tight_layout()
        self.save_img(plt, "ICA Analisys" + "(" + str(1) + ")")
        plt.close()
