from preprocessing.Preprocessing import Preprocessing
from processing.Processing import Processing
import pandas as pd
import os
from utils.Saving import Saving
from utils.Optimizing import Optimizing
from utils.Models import Models


def make_preprocessing(
    url: str,
    column_filter: str,
    values: list,
    variable_delete: list,
    wsize=8,
):
    prp = Preprocessing()
    filepath = prp.download_dataset(url, exist_ok=True)

    data = prp.read_dataset(filepath, header=0, index_col=0)
    data = prp.filter_by_names(data, column_filter, values)
    data = prp.eliminate_variables(data, variable_delete, axis=1)
    prp.get_plots(data)
    prp.get_PCA(data)
    prp.get_ICA(data)
    prp.get_correlation_matrix(data)

    return data, prp


def slide_data(data, prp: object, wsize, t_label="UCI"):
    data = prp.window_slide_dataset(data, wsize, wsize - 1)
    X, t = Preprocessing.split_data(data, t_label)
    return X, t


def make_processing(X, t, windowsize):
    models = Models()
    training_results = {}
    for wsize in windowsize:
        X, t = slide_data(data, prp, wsize)
        processing = Processing(X, t, wsize)
        processing.perform_optimizing_model(
            models.models, return_train_score=True
        )
        training_results[wsize] = processing.perform_training_model(
            models.models
        )
        processing.perform_validation_curve(models.models)
        processing.perform_plot_predictions(models.models)
    processing.perform_wsize_comparison(
        training_results, models.models, windowsize
    )


if __name__ == "__main__":
    data, prp = make_preprocessing(
        "https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/provincias_covid19_datos_sanidad_nueva_serie.csv",
        "provincia",
        ["A Coruña", "Lugo", "Ourense", "Pontevedra"],
        ["cod_ine"],
    )

    make_processing(data, prp, [2, 3, 4])
