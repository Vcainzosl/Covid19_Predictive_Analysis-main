from utils.Saving import Saving
from utils.Models import Models
from utils.Splitter import Splitter
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


class Modeling:
    def __init__(self, X, t, model, mae=True, mse=True):
        self.model = model
        self.predict_model(X, t)
        self.get_metrics(t, mae, mse)

    def predict_model(self, X, t):
        self.y_pred = self.model.best_model.predict(X)

    def get_metrics(self, t, mae, mse):
        self.metrics = {"Model": self.model.model.model_name}
        if mae:
            self.metrics["MAE"] = mean_absolute_error(t, self.y_pred)
        if mse:
            self.metrics["MSE"] = mean_squared_error(t, self.y_pred)
    