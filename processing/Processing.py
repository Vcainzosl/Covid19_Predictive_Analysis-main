from utils.Optimizing import Optimizing
from utils.Saving import Saving
from utils.Modeling import Modeling
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Processing(Saving):
    def __init__(self, X, t, wsize, train_size=0.8):
        super().__init__()
        self.X_train, self.X_test, self.t_train, self.t_test = train_test_split(
            X, t, train_size=0.8, shuffle=False
        )
        self.X = X
        self.t = t
        self.wsize = wsize

    def perform_optimizing_model(self, models: list, **kwargs):
        self.model_optimizing = {}
        for model in models:
            self.model_optimizing[model.model_name] = Optimizing(
                self.X_train,
                self.t_train,
                model,
                **kwargs,
            )
            self.save_csv(
                pd.DataFrame(
                    self.model_optimizing[model.model_name].best_results
                )
                .round(4)
                .to_csv(index=False),
                "Optimization "
                + model.model_name
                + "-windowsize="
                + str(self.wsize),
            )

            self.save_csv(
                pd.DataFrame.from_records(
                    [self.model_optimizing[model.model_name].best_params]
                )
                .round(4)
                .to_csv(index=False),
                "Best hyperparameters "
                + model.model_name
                + "-windowsize="
                + str(self.wsize),
            )

    def perform_optimizing_DNN():
        DNN_optimizing = DNNOptimizing()

    def perform_training_model(self, models: list, **kwargs):
        self.model_training = {}
        for model in models:
            self.model_training[model.model_name] = Modeling(
                self.X_test,
                self.t_test,
                self.model_optimizing[model.model_name],
                **kwargs,
            )

            self.save_csv(
                pd.DataFrame.from_records(
                    [self.model_training[model.model_name].metrics]
                )
                .round(4)
                .to_csv(index=False),
                "Metrics evaluation "
                + model.model_name
                + "-windowsize="
                + str(self.wsize),
            )
        return self.model_training

    def perform_plot_predictions(self, models: list):
        for model in models:
            fig = plt.figure(figsize=(10, 5))
            ind = pd.to_datetime(self.t_test.index)
            plt.title(model.model_name + " - " + self.t.name)
            plt.plot(ind, self.t_test, label="Real")
            plt.plot(
                ind,
                self.model_training[model.model_name].y_pred,
                label="Prediction",
            )
            plt.legend()
            self.save_img(
                plt,
                "Prediction "
                + model.model_name
                + "-"
                + self.t.name
                + ", windowsize="
                + str(self.wsize),
            )

    def perform_validation_curve(self, models: list):
        for model in models:
            df = pd.DataFrame(
                self.model_optimizing[model.model_name].best_results
            )
            results = ["mean_test_score", "mean_train_score"]
            for (param_name, param_range) in self.model_optimizing[
                model.model_name
            ].params_dist.items():
                grouped_df = df.groupby(f"param_{param_name}")[results].agg(
                    {
                        "mean_train_score": "mean",
                        "mean_test_score": "mean",
                    }
                )
                fig = plt.figure(figsize=(10, 5))
                plt.xlabel(param_name)
                plt.title(model.model_name)
                plt.plot(
                    param_range,
                    grouped_df["mean_train_score"],
                    label="Training score",
                    color="darkorange",
                    marker="o",
                )
                plt.plot(
                    param_range,
                    grouped_df["mean_test_score"],
                    label="Cross-validation score",
                    color="navy",
                    marker="o",
                )
                plt.legend()
                self.save_img(
                    plt,
                    "Validation curve "
                    + model.model_name
                    + "-"
                    + param_name
                    + ", windowsize="
                    + str(self.wsize),
                )

    def perform_wsize_comparison(
        self, results: dict, models, windowsize, scoring=["MAE", "MSE"]
    ):

        for model in models:
            fig = plt.figure(figsize=(10, 5))
            plt.title(model.model_name)
            model_metrics = []
            for wsize in windowsize:
                df = pd.DataFrame(
                    results[wsize][model.model_name].metrics, index=["Metrics"]
                )
                model_metrics.append(df[scoring].values)
            model_metrics = np.asarray(model_metrics)
            windowsize = np.asarray(windowsize)
            for i in model_metrics.T:
                plt.plot(windowsize, i[0], label=scoring[0], marker="o")
            plt.legend()
            plt.xlabel("Windowsize")
            plt.xticks(windowsize)
            self.save_img(plt, "Windowsize comparison " + model.model_name)
