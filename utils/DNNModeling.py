import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class DNNModeling:
    """Class for predicting values of DNN model and get metrics"""

    def __init__(self, X, t, X_pred, DNN, mae=True, mse=True):
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
        self.DNN = DNN
        # Call predict method
        self.predict_DNN(X, t, X_pred)
        # Get metrics
        self.get_metrics(t, mae, mse)

    def predict_DNN(self, X, t, X_pred):
        """Performs prediction values

        :param X: matrix test samples
        :type X: pandas.DataFrame
        :param t: array test samples
        :type t: pandas.Series
        :param X_pred: matrix future values
        :type X_pred: pandas.DataFrame
        """
        # Predict values with labels
        self.y = self.DNN.predict(X)
        # Concat future values predictions
        self.y_pred = self.DNN.predict(pd.concat([X, X_pred]))

    def get_metrics(self, t, mae, mse):
        """Calculate error metrics

        :param t: array with real values
        :type t: pandas.Series
        :param mae: calculate MAE
        :type mae: bool
        :param mse: calculate MSE
        :type mse: bool
        """
        self.metrics = {"Model": "DNN"}
        if mae:
            self.metrics["MAE"] = mean_absolute_error(t.values, self.y)
        if mse:
            self.metrics["MSE"] = mean_squared_error(t.values, self.y)
