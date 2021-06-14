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
        """Contructor of Preprocessing class

        :param base_dir: base directory, defaults to "."
        :type base_dir: str, optional
        :param csv: create csv folder, defaults to True
        :type csv: bool, optional
        :param img: create img folder, defaults to True
        :type img: bool, optional
        :param pickle: create pickle folder, defaults to False
        :type pickle: bool, optional
        :param input_folder: input folder to save base Dataset, defaults to "Datasets"
        :type input_folder: str, optional
        """
        # Inheritance of Saving class
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
        filename="Original-Data",
        save=True,
        **kwargs,
    ) -> pd.DataFrame:
        """Reads Dataset from csv file

        :param filepath: path of the csv file
        :type filepath: str
        :param filename: name of the new csv file, defaults to "Original Data"
        :type filename: str, optional
        :param save: save the new file, defaults to True
        :type save: bool, optional
        :return: dataset read
        :rtype: pd.DataFrame
        """
        data = pd.read_csv(filepath, **kwargs)
        if save:
            self.save_csv(data.head().to_csv(), filename)
        return data

    def filter_by_names(
        self,
        data: pd.DataFrame,
        column: str,
        values: list,
        filename="Filter-Data",
        save=True,
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
        :param save: save the new file, defaults to True
        :type save: bool, optional
        :return: DataSet filtered
        :rtype: pd.DataFrame
        """
        data = data[data[column].isin(values)]
        data = data.groupby(data.index.name, **kwargs)[data.columns].sum()
        if save:
            self.save_csv(data.head().to_csv(), filename)
        return data

    def eliminate_variables(
        self,
        data: pd.DataFrame,
        columns: list,
        filename="Elimination-Data",
        save=True,
        **kwargs,
    ) -> pd.DataFrame:
        """delete those columns in DataSet unwanted

        :param data: Dataset with samples
        :type data: pd.DataFrame
        :param columns: names of variables to delete
        :type columns: list
        :param filename: name of the DataSet after elimination .csv file
        :type filename: str
        :param save: save the new file, defaults to True
        :type save: bool, optional
        :return: Dataset clean of undesired features
        :rtype: pd.DataFrame
        """
        data = data.drop(columns, **kwargs)
        if save:
            self.save_csv(data.head().to_csv(), filename)
        # Save cleaned dataset as instance attribute
        self.dataset = data
        return data

    def window_slide_dataset(
        self,
        window,
        prediction,
        label="Casos",
        wrapper_class=WindowData,
        filename="Slided-Data",
        save=True,
        **kwargs,
    ) -> pd.DataFrame:
        """Runs sliding window with overlapping on DataSet values

        :param window: number of past samples (from today)
        :type window: int
        :param prediction: number of future sample
        :type prediction: int
        :param label: label to predict, defaults to "Casos"
        :type label: str, optional
        :param wrapper_class: wrapper class to build new slided dataset, defaults to WindowData
        :type wrapper_class: Class, optional
        :param filename: name of the new file, defaults to "Slided Data"
        :type filename: str, optional
        :param save: save new csv file, defaults to True
        :type save: bool, optional
        :return: matrix with samples, array with labels and matrix with samples to predict new values
        :rtype: pandas.DataFrame, pandas.Series, pandas.DataFrame
        """

        window = window
        prediction = prediction
        bucket_size = window + prediction + 1
        overlap_count = bucket_size - 1

        samples = []
        columns = []
        # Use each variable of dataset
        for name in self.dataset.columns:
            for i in range(bucket_size):
                # Past samples
                if i < window:
                    columns.append(name + " t" + str(i - window))
                # Current sample
                elif i == window:
                    columns.append(name)
                # Future samples
                elif i > window:
                    columns.append(name + " t+" + str(i - window))

        slider = Slider(bucket_size, overlap_count, WindowData)
        # Using dataset trapose
        slider.fit(self.dataset.T.values)
        while True:
            window_data = slider.slide()
            if slider.reached_end_of_list():
                break
            samples.append(window_data.build_sample())

        # Completed data slided
        Data = pd.DataFrame(
            samples,
            columns=columns,
            index=self.dataset.T.columns[bucket_size - 1 :],
        )
        # Data for target label
        Mlabel = Data[[column for column in Data.columns if (label in column)]]
        # Values without labels to predict
        l = Mlabel[-1:].values[0]
        rows = []
        bucket_size = window + 1
        overlap_count = bucket_size - 1
        slider = Slider(bucket_size, overlap_count)
        # Slide values
        slider.fit(l.T)
        while True:
            window_data = slider.slide()
            rows.append(window_data)
            if slider.reached_end_of_list():
                break
        xcol = [
            column
            for column in Data.columns
            if ("+" not in column) & (label in column)
        ]

        # Samples to predict output values, data without labels to train for
        predict = pd.DataFrame(rows, columns=xcol).iloc[1:-1]
        ind = [
            pd.to_datetime(Mlabel[-1:].index)[0] + pd.Timedelta(days=i)
            for i in predict.index
        ]
        predict = predict.set_index([ind])

        # Matrix of samples, array of labels and matrix of samples to predict (without labels)
        X = Data[xcol]
        t = Data[label + " t+" + str(prediction)]
        if save:
            self.save_csv(
                X.head().to_csv(),
                "X-windowsize="
                + str(window)
                + ",-"
                + label
                + "-t+"
                + str(prediction),
            )
            self.save_csv(
                t.head().to_csv(),
                "t-windowsize="
                + str(window)
                + ",-"
                + label
                + "-t+"
                + str(prediction),
            )

        return X, t, predict

    def get_plots(self, data: pd.DataFrame, filename="Distribution-Data"):
        """Plot original data distibutions

        :param data: original dataset
        :type data: pandas.DataFrame
        :param filename: name of img file, defaults to "Distribution Data"
        :type filename: str, optional
        """
        fig = plt.figure(len(data.columns), figsize=(10, 5))
        ind = pd.to_datetime(data.index.tolist())
        plt.plot(ind[:], data.iloc[:, 0].values)
        plt.title(data.columns[0])
        plt.legend([data.columns[0]], loc="upper right")
        plt.tight_layout()
        self.save_img(plt, filename)

        for i in range(len(data.columns) - 1):
            plt.clf()
            plt.plot(ind[:], data.iloc[:, i + 1].values)
            plt.title(data.columns[i + 1])
            plt.legend([data.columns[i + 1]], loc="upper right")
            plt.tight_layout()
            # Save each plot as img file
            self.save_img(plt, filename + "(" + str(i + 1) + ")")
        plt.close()

    def get_correlation_matrix(self, X, filename="Correlation-matrix"):
        """Method to get correlation matrix

        :param X: matrix with inputs values (variables)
        :type X: pandas.DataFrame
        :param filename: name of the img file, defaults to "Correlation matrix"
        :type filename: str, optional
        """
        samples, nvar = X.shape
        plt.figure(figsize=(7, 6))
        # Correlation matrix using all variables
        corr_mat = np.corrcoef(np.c_[X].T)
        etiquetas = X.columns.values.tolist()
        # Heat map over correlation matrix variables
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
        self.save_img(plt, filename)
        plt.close()

    def get_PCA(self, X, filename="PCA-analisys", **kwargs):
        """Performs PCA analysis

        :param X: matrix with input variables
        :type X: pandas.DataFrame
        :param filename: name of img file, defaults to "PCA analisys"
        :type filename: str, optional
        """
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
        self.save_img(plt, filename)
        plt.close()
