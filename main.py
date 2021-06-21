from preprocessing.Preprocessing import Preprocessing
from processing.Processing import Processing
from postprocessing.Postprocessing import Postprocessing
from utils.Models import Models


def make_preprocessing(
    url: str,
    column_filter: str,
    values: list,
    variable_delete: list,
):
    """Main function to perform preprocessing tasks

    :param url: dataset url
    :type url: str
    :param column_filter: column name to filter values
    :type column_filter: str
    :param values: values to filter samples
    :type values: list
    :param variable_delete: delete variable of the dataset
    :type variable_delete: list
    :return: Preprocessing object
    :rtype: class Preprocessing
    """
    prp = Preprocessing()
    filepath = prp.download_dataset(url, exist_ok=True)

    data = prp.read_dataset(filepath, header=0, index_col=0)
    data = prp.filter_by_names(data, column_filter, values)
    data = prp.eliminate_variables(data, variable_delete, axis=1)
    prp.get_plots(data)
    prp.get_correlation_matrix(data)
    prp.get_PCA(data)

    return prp


def slide_data(prp: object, wsize, prediction, t_label="Casos"):
    """Performs data sliding to an specific windowsize and prediction

    :param prp: Preprocesing object
    :type prp: class Preprocessing
    :param wsize: number of past samples
    :type wsize:  int
    :param prediction: number of future samples
    :type prediction: int
    :param t_label: label to predict, defaults to "Casos"
    :type t_label: str, optional
    :return: Matrix with samples, array with labels and matrix with samples to predict future values
    :rtype: pandas.DataFrame, pandas.Series, pandas.DataFrame
    """
    X, t, X_pred = prp.window_slide_dataset(wsize, prediction)
    return X, t, X_pred


def make_processing(prp, windowsize):
    """Main function to perform processing calling specific methods

    :param prp: Preprocessing object
    :type prp: class Preprocessing
    :param windowsize: range of windosize
    :type windowsize: list
    """
    # Create object Models
    models = Models()
    results = {}
    DNNscores = {}
    cv = int(input("Número de K-folds: "))
    trials = int(input("Número de intentos por modelo: "))
    epochs = int(input("Número de epochs: "))
    batch_size = int(input("Muestras por lote (batch): "))
    # Predictions same as windosize
    predictions = windowsize
    for wsize in windowsize:
        testing_results = {}
        # Same values to predict as windowsize
        for prediction in predictions:
            X, t, X_pred = slide_data(prp, wsize, prediction)
            # Create object Processing
            processing = Processing(
                X, t, X_pred, wsize, cv, trials, epochs, batch_size
            )
            # Save DNN training results to plot validation curve
            DNNscores[str(prediction)] = processing.perform_optimizing_model(
                models.models,
                return_train_score=True,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            # Save test results for each model to plot windowsize comparison
            testing_results[str(prediction)] = processing.perform_testing_model(
                models.models
            )
            # Plot mean scores of training for each model
            processing.perform_validation_models(models.models)
            # Plot predictions for each model
            processing.perform_plot_predictions(models.models)
        # Plot DNN validation curve for each prediction and current windowsize
        processing.perform_validation_DNN(DNNscores)
        # Save predictions results for current windowsize
        results[wsize] = testing_results
    processing.perform_wsize_comparison(
        results, predictions, models.models, windowsize
    )
    processing.perform_prediction_comparison(
        results, predictions, models.models, windowsize
    )


def make_postprocessing(filename, **kwargs):
    """Main function to elaborate the report containing all simulation results

    :param filename: name of the report file
    :type filename: str
    """
    pop = Postprocessing(filename, **kwargs)


if __name__ == "__main__":

    prp = make_preprocessing(
        "https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/provincias_covid19_datos_sanidad_nueva_serie.csv",
        "provincia",
        ["A Coruña", "Lugo", "Ourense", "Pontevedra"],
        ["cod_ine"],
    )

    make_processing(prp, [1, 7, 14])
    # In case it is wanted, with the line below uncommented, a report is generated on the base directory
    # make_postprocessing("report", title="Report", author="Víctor")
