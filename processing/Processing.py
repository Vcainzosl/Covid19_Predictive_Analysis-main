from utils.Optimizing import Optimizing
from utils.Saving import Saving
from utils.Modeling import Modeling
from utils.DNNOptimizing import DNNOptimizing
from utils.DNNModeling import DNNModeling
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Processing(Saving):
    """Inheritance of Saving class

    :param Saving: class Saving, with util methods
    :type Saving: class
    """

    def __init__(
        self,
        X,
        t,
        X_pred,
        wsize,
        cv,
        trials,
        epochs,
        batch_size,
        train_size=0.8,
    ):
        """Constructor of Processing class

        :param X: matrix of samples
        :type X: pandas.DataFrame
        :param t: array of labels
        :type t: pandas.Series
        :param X_pred: matrix with samples to predict
        :type X_pred: pandas.Dataframe
        :param wsize: windosize (number of past samples)
        :type wsize: int
        :param cv: number of kfold for cross validation
        :type cv: int
        :param trials: number of trials per model
        :type trials: int
        :param epochs: number of epochs for DNN
        :type epochs: int
        :param train_size: size of training set, defaults to 0.8
        :type train_size: float, optional
        """
        super().__init__()
        # Split dataset in training and test sets
        self.X_train, self.X_test, self.t_train, self.t_test = train_test_split(
            X, t, train_size=0.8, shuffle=False
        )
        # Shuffle training set to avoid overfitting but not test set to predict historical data
        train = pd.DataFrame.copy(self.X_train)
        train[self.t_train.name] = self.t_train.values
        shuffle = train.sample(frac=1)
        self.t_train = shuffle[self.t_train.name]
        self.X_train = shuffle.drop(self.t_train.name, axis=1)

        self.t = t
        self.X_pred = X_pred
        self.wsize = wsize
        self.cv = cv
        self.trials = trials
        self.epochs = epochs
        self.batch_size = batch_size

    def perform_optimizing_model(
        self, models: list, filename="Best-hyperparameters", **kwargs
    ):
        """Performs optimizing models

        :param models: models of Scikit-Learn library
        :type models: list
        :param filename: name of the output csv file, defaults to "Best hyperparameters"
        :type filename: str, optional
        :return: DNNscores to plot validation curve
        :rtype: dict
        """
        # Dictionary to include Optiming objects for each model
        self.model_optimizing = {}
        # List to append best hyperparameter for each model
        results = []
        for model in models:
            self.model_optimizing[model.model_name] = Optimizing(
                self.X_train,
                self.t_train.ravel(),
                model,
                self.cv,
                self.trials,
                **kwargs,
            )
            results.append(self.model_optimizing[model.model_name].best_params)

        DNNscores = self.perform_optimizing_DNN()

        # Add DNN
        results.append(self.DNN_optimizing.params)

        # Save best hyperparameters as DataFrame
        self.save_csv(
            pd.DataFrame(results).round(4).fillna("-").to_csv(index=False),
            filename + "-windowsize=" + str(self.wsize) + ",-" + self.t.name,
        )
        # Return validation and training scores for DNN to plot validation curve
        return DNNscores

    def perform_optimizing_DNN(self):
        """Performs DNN optiming specifically

        :return: validation adn training scores
        :rtype: dict
        """
        self.DNN_optimizing = DNNOptimizing(
            self.X_train,
            self.t_train,
            self.cv,
            self.trials,
            self.epochs,
            self.batch_size,
        )
        return self.DNN_optimizing.scores

    def perform_testing_model(
        self, models: list, filename="Metrics-evaluation", **kwargs
    ):
        """Performs testing models

        :param models: models of scikit-learn library
        :type models: list
        :param filename: name of the output csv file, defaults to "Metrics evaluation"
        :type filename: str, optional
        :return: dictionary with Modeling objects
        :rtype: dict
        """
        # Dictionary for Modeling objects
        self.model_testing = {}
        # List to append metrics for each model
        results = []
        for model in models:
            self.model_testing[model.model_name] = Modeling(
                self.X_test,
                self.t_test.ravel(),
                self.X_pred,
                self.model_optimizing[model.model_name],
                **kwargs,
            )
            results.append(self.model_testing[model.model_name].metrics)

        # Call method of DNN
        self.perform_testing_DNN()

        # Add DNN metric results
        results.append(self.DNN_modeling.metrics)
        self.model_testing["DNN"] = self.DNN_modeling

        # Build a DataFrame and save it as csv file
        self.save_csv(
            pd.DataFrame(results).fillna("-").round(4).to_csv(index=False),
            filename + "-windowsize=" + str(self.wsize) + ",-" + self.t.name,
        )
        return self.model_testing

    def perform_testing_DNN(self):
        """Performs testing of DNN model"""
        self.DNN_modeling = DNNModeling(
            self.X_test, self.t_test, self.X_pred, self.DNN_optimizing.bestDNN
        )

    def perform_plot_predictions(self, models: list, colors=["b", "y"]):
        """Plots predicted outputs for each model

        :param models: models of scikit-learn
        :type models: list
        :param colors: color for plotting, defaults to ["b", "y"]
        :type colors: list, optional
        """

        fig = plt.figure(figsize=(10, 5))
        # Index of data test
        ind = pd.to_datetime(self.X_test.index)
        # Index of prediction samples
        indp = pd.to_datetime(pd.concat([self.X_test, self.X_pred]).index)
        plt.title("Windowsize " + str(self.wsize) + ", " + self.t.name)
        # Plot real outputs
        plt.plot(
            ind,
            self.t_test,
            label=self.t.name,
            linestyle="dashed",
        )
        # Plot predicted outputs
        for model, color in zip(models, colors):
            plt.plot(
                indp,
                self.model_testing[model.model_name].y_pred,
                label="Prediction-" + model.model_name,
                color=color,
            )

        # Plot DNN
        plt.plot(
            indp,
            self.model_testing["DNN"].y_pred,
            label="Prediction-DNN",
            color="g",
        )
        # Line defining the date until the model could predict values
        plt.axvline(
            indp[-1],
            color="r",
            linestyle="dotted",
        )
        plt.text(
            indp[-2],
            np.mean(self.t_test.values),
            str(indp[-1]).split(" ")[0],
            rotation="vertical",
            bbox=dict(boxstyle="round", fc="0.8"),
        ),
        plt.legend()

        # Save predictions as img file
        self.save_img(
            plt,
            "Prediction-"
            + ("-").join(self.t.name.split(" "))
            + ",-windowsize="
            + str(self.wsize),
        )
        plt.close()

    def perform_validation_models(
        self, models: list, filename="Validation-curve", colors=["b", "y"]
    ):
        """Plots mean validation results for models

        :param models: models of scikit-learn library
        :type models: list
        :param filename: name of the output img file, defaults to "Validation curve"
        :type filename: str, optional
        :param colors: colors for plotting, defaults to ["b", "y"]
        :type colors: list, optional
        """

        fig = plt.figure(figsize=(10, 5))
        plt.title("Windowsize=" + str(self.wsize) + ", " + self.t.name)
        # Create index with number of kfolds
        ind = np.arange(1, self.cv + 1)
        # Labels for validation average results
        nmodels = np.arange(1, len(models) + 1)
        # Plot validation results for each model
        for model, color, n in zip(models, colors, nmodels):
            results = self.model_optimizing[model.model_name]
            train = []
            val = []
            for c in results.best_results:
                if "split" in c and "train" in c:
                    train.append(results.best_results[c][results.best_index])
                elif "split" in c and "test" in c:
                    val.append(results.best_results[c][results.best_index])
            plt.xlabel("cv")
            plt.ylabel("MSE")
            # Mean validation results
            plt.axhline(
                np.mean(-np.asarray(val)),
                color=color,
                linestyle="dotted",
            )
            plt.text(
                n,
                np.mean(-np.asarray(val)) + 1,
                "Media: " + str(int(np.mean(-np.asarray(val)) + 1)),
                bbox=dict(boxstyle="round", fc="0.8"),
            )
            plt.plot(
                ind,
                -np.asarray(val),
                label="Val-" + model.model_name,
                marker="o",
                color=color,
            )
            plt.plot(
                ind,
                -np.asarray(train),
                label="Train-" + model.model_name,
                marker="x",
                color=color,
                linewidth=1,
                linestyle="dashed",
            )
        # Add DNN results
        plt.axhline(
            np.mean(np.mean(self.DNN_optimizing.scores["val"], axis=1)),
            color="g",
            linestyle="dotted",
        )
        plt.text(
            n + 1,
            np.mean(np.mean(self.DNN_optimizing.scores["val"], axis=1)),
            "Media: "
            + str(
                int(
                    np.mean(np.mean(self.DNN_optimizing.scores["val"], axis=1))
                    + 1
                )
            ),
            bbox=dict(boxstyle="round", fc="0.8"),
        )
        plt.plot(
            ind,
            np.mean(self.DNN_optimizing.scores["val"], axis=1),
            label="Val-DNN",
            marker="o",
            c="g",
        )
        plt.plot(
            ind,
            np.mean(self.DNN_optimizing.scores["train"], axis=1),
            label="Train-DNN",
            marker="x",
            c="g",
            linewidth=1,
            linestyle="dashed",
        )

        plt.legend()
        plt.xticks(np.arange(1, self.cv + 1))
        plt.tight_layout()
        # Save output img file
        self.save_img(
            plt,
            filename
            + "-windowsize="
            + str(self.wsize)
            + ",-"
            + ("-").join(self.t.name.split(" ")),
        )
        plt.close()

    def perform_validation_DNN(self, DNNscores, colors=["r", "c", "m"]):
        """Plot DNN validation curve

        :param DNNscores: validation and training scores for DNN
        :type DNNscores: dict
        :param colors: colors for plotting, defaults to ["r", "c", "m"]
        :type colors: list, optional
        """
        fig = plt.figure(figsize=(10, 5))
        plt.title("Windowsize_" + str(self.wsize) + ", DNN")
        # Define index with number of epochs
        ind = np.arange(1, self.DNN_optimizing.epochs + 1)
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        # Lenght of scores to plot labels
        size = np.arange(len(DNNscores))
        for i, color, point in zip(DNNscores, colors, size):
            yval = np.mean(DNNscores[i]["val"], axis=0)
            ytrain = np.mean(DNNscores[i]["train"], axis=0)
            plt.plot(
                ind,
                np.mean(DNNscores[i]["val"], axis=0),
                label="Val-DNN, " + self.t.name.split(" ")[0] + " t+" + i,
                c=color,
            )
            plt.plot(
                ind,
                np.mean(DNNscores[i]["train"], axis=0),
                label="Train-DNN, " + self.t.name.split(" ")[0] + " t+" + i,
                linestyle="dashed",
                c=color,
            )
            plt.annotate(
                "%i" % (yval[-1]),
                (len(yval) - 1, yval[-1]),
                xytext=(len(yval) + (point - 2), yval[-2]),
                bbox=dict(boxstyle="round", fc="0.8"),
            )
        plt.legend()
        plt.xticks(np.arange(0, self.DNN_optimizing.epochs + 1, 2))
        plt.tight_layout()
        # Save output img
        self.save_img(plt, "Validation-DNN-windowsize=" + str(self.wsize))
        plt.close()

    def perform_wsize_comparison(
        self,
        results: dict,
        predictions: list,
        models,
        windowsize,
        scoring=["MAE", "MSE"],
        colors=["b", "y"],
    ):
        """Plots metrics for each model and windowsize

        :param results: results with Modeling objects for each model
        :type results: dict
        :param t: predictions dates from today
        :type t: list
        :param models: models of scikit-learn
        :type models: list
        :param windowsize: windowsize, number of past samples considered as variables
        :type windowsize: int
        :param scoring: metrics used, defaults to ["MAE", "MSE"]
        :type scoring: list, optional
        :param colors: colors for plotting, defaults to ["b", "y"]
        :type colors: list, optional
        """
        # Plot windowsize for each prediction, score and model
        # Labels for metrics average results
        windows = np.arange(0, len(windowsize))
        for n in predictions:
            label = str(n)
            for score in scoring:
                fig = plt.figure(figsize=(10, 5))
                plt.title(self.t.name.split(" ")[0] + " t+" + label)
                for model, color, n in zip(models, colors, windows):
                    model_metrics = []
                    for wsize in windowsize:
                        df = pd.DataFrame(
                            results[wsize][label][model.model_name].metrics,
                            index=[label],
                        )
                        model_metrics.append(df[score].values)
                    model_metrics = np.asarray(model_metrics)
                    windowsize = np.asarray(windowsize)
                    plt.plot(
                        windowsize,
                        model_metrics,
                        label=model.model_name,
                        marker="o",
                    )
                    plt.axhline(
                        np.mean(np.mean(model_metrics, axis=1)),
                        color=color,
                        linestyle="dotted",
                    )
                    plt.text(
                        windowsize[n],
                        np.mean(model_metrics) + 1,
                        "Media: " + str(int(np.mean(model_metrics) + 1)),
                        bbox=dict(boxstyle="round", fc="0.8"),
                    )
                    plt.legend()
                    plt.xlabel("Windowsize")
                    plt.xticks(windowsize)

                # Add DNN results
                DNN_metrics = []
                for wsize in windowsize:
                    dfDNN = pd.DataFrame(
                        results[wsize][label]["DNN"].metrics,
                        index=[label],
                    )
                    DNN_metrics.append(dfDNN[score].values)

                DNN_metrics = np.asarray(DNN_metrics)
                windowsize = np.asarray(windowsize)
                plt.plot(
                    windowsize,
                    DNN_metrics,
                    label="DNN",
                    marker="o",
                )
                plt.axhline(
                    np.mean(np.mean(DNN_metrics, axis=1)),
                    color="g",
                    linestyle="dotted",
                )
                plt.text(
                    windowsize[n + 1],
                    np.mean(DNN_metrics) + 1,
                    "Media: " + str(int(np.mean(DNN_metrics) + 1)),
                    bbox=dict(boxstyle="round", fc="0.8"),
                )
                plt.legend()
                plt.xlabel("Windowsize")
                plt.ylabel(score)
                plt.xticks(windowsize)

                # Save img file
                self.save_img(
                    plt,
                    "Windowsize-comparison-" + score + ",-t+" + label,
                )
                plt.close()

    def perform_prediction_comparison(
        self,
        results: dict,
        predictions: list,
        models,
        windowsize,
        scoring=["MAE", "MSE"],
        colors=["b", "y"],
    ):
        """Plots metrics for each model and prediction

        :param results: results with Modeling objects for each model
        :type results: dict
        :param t: predictions dates from today
        :type t: list
        :param models: models of scikit-learn
        :type models: list
        :param windowsize: windowsize, number of past samples considered as variables
        :type windowsize: int
        :param scoring: metrics used, defaults to ["MAE", "MSE"]
        :type scoring: list,
        :param colors: colors for plotting, defaults to ["b", "y"]
        :type colors: list, optional
        """
        # Plot prediction for each windowsize, score and model
        # Labels for metrics average results
        predicts = np.arange(0, len(predictions))
        for wsize in windowsize:
            for score in scoring:
                fig = plt.figure(figsize=(10, 5))
                plt.title("Windowsize=" + str(wsize))
                for model, color, n in zip(models, colors, predicts):
                    model_metrics = []
                    for prediction in predictions:
                        label = str(prediction)
                        df = pd.DataFrame(
                            results[wsize][label][model.model_name].metrics,
                            index=[label],
                        )
                        model_metrics.append(df[score].values)
                    model_metrics = np.asarray(model_metrics)
                    predictions = np.asarray(predictions)
                    plt.plot(
                        predictions,
                        model_metrics,
                        label=model.model_name,
                        marker="o",
                    )
                    plt.axhline(
                        np.mean(np.mean(model_metrics, axis=1)),
                        color=color,
                        linestyle="dotted",
                    )
                    plt.text(
                        predictions[n],
                        np.mean(model_metrics) + 1,
                        "Media: " + str(int(np.mean(model_metrics) + 1)),
                        bbox=dict(boxstyle="round", fc="0.8"),
                    )
                    plt.legend()
                    plt.xlabel("Predictions")
                    plt.ylabel(score)
                    plt.xticks(predictions)

                # Add DNN results
                DNN_metrics = []
                for prediction in predictions:
                    label = str(prediction)
                    dfDNN = pd.DataFrame(
                        results[wsize][label]["DNN"].metrics,
                        index=[label],
                    )
                    DNN_metrics.append(dfDNN[score].values)

                DNN_metrics = np.asarray(DNN_metrics)
                predictions = np.asarray(predictions)
                plt.plot(
                    predictions,
                    DNN_metrics,
                    label="DNN",
                    marker="o",
                )
                plt.axhline(
                    np.mean(np.mean(DNN_metrics, axis=1)),
                    color="g",
                    linestyle="dotted",
                )
                plt.text(
                    predictions[n + 1],
                    np.mean(DNN_metrics) + 1,
                    "Media: " + str(int(np.mean(DNN_metrics) + 1)),
                    bbox=dict(boxstyle="round", fc="0.8"),
                )
                plt.legend()
                plt.xlabel("Predictions")
                plt.xticks(predictions)

                # Save img file
                self.save_img(
                    plt,
                    "Predictions-comparison-"
                    + score
                    + ",-windowsize="
                    + str(wsize),
                )
                plt.close()
