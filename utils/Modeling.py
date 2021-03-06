import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Modeling:
    """Class for predicting values of models and calculate error metrics"""

    def __init__(self, X, t, X_pred, model, mae=True, mse=True):
        """Constructor of class DNNModeling

        :param X: matrix with test samples
        :type X: pandas.DataFrame
        :param t: array with test labels
        :type t: pandas.Series
        :param X_pred: matrix with samples to predict
        :type X_pred: pandas.DataFrame
        :param DNN: best DNN model
        :type DNN: keras model
        :param mae: calculate mae metric, defaults to True
        :type mae: bool, optional
        :param mse: calculate mse metric, defaults to True
        :type mse: bool, optional
        """
        self.model = model
        self.predict_model(X, t, X_pred)
        self.get_metrics(t, mae, mse)

    def predict_model(self, X, t, X_pred):
        """Performs prediction values

        :param X: matrix test samples
        :type X: pandas.DataFrame
        :param t: array test samples
        :type t: pandas.Series
        :param X_pred: matrix future values
        :type X_pred: pandas.DataFrame
        """
        self.y = self.model.best_model.predict(X)
        self.y_pred = self.model.best_model.predict(pd.concat([X, X_pred]))

    def get_metrics(self, t, mae, mse):
        """Calculate error metrics

        :param t: array with real values
        :type t: pandas.Series
        :param mae: calculate MAE
        :type mae: bool
        :param mse: calculate MSE
        :type mse: bool
        """
        self.metrics = {"Model": self.model.model.model_name}
        if mae:
            self.metrics["MAE"] = mean_absolute_error(t, self.y)
        if mse:
            self.metrics["MSE"] = mean_squared_error(t, self.y)
