from tensorflow import keras
from sklearn.model_selection import KFold
import random
import numpy as np


class DNNOptimizing:
    """Class implemeted to optimize DNN model"""

    def __init__(
        self,
        X,
        t,
        cv,
        trials,
        epochs,
        batch_size,
        layers=(1, 5),
        n=(10, 20),
        activation="relu",
        optimizer="adam",
        loss="mean_squared_error",
    ):
        """Contructor of DNNOptiming class

        :param X: matrix of training samples
        :type X: pandas.DataFrame
        :param t: array of training labels
        :type t: pandas.Series
        :param cv: number of kfolds
        :type cv: int
        :param trials: number of trials per DNN
        :type trials: int
        :param epochs: number of epochs
        :type epochs: int
        :param layers: range of layers, defaults to (1, 5)
        :type layers: tuple, optional
        :param n: range of neurons, defaults to (10, 20)
        :type n: tuple, optional
        :param activation: activation function, defaults to "relu"
        :type activation: str, optional
        :param optimizer: optimizer method used, defaults to "adam"
        :type optimizer: str, optional
        :param loss: loss function used, defaults to "mean_squared_error"
        :type loss: str, optional
        """
        self.X = X
        self.t = t
        self.layers = layers
        self.n = n
        self.activation = activation
        self.loss = loss
        # Call optimizing method
        self.optimize_DNN(cv, trials, epochs, batch_size)
        # Get results
        self.get_best_hyperparameters()

    def create_random_network(self):
        """Generates a random network according to hyperparameters range values

        :return: DNN model
        :rtype: Keras model
        """
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(
                random.randint(self.n[0], self.n[1]),
                input_dim=self.X.shape[1],
                activation=self.activation,
            )
        )
        for _ in range(random.randint(self.layers[0], self.layers[1])):
            model.add(
                keras.layers.Dense(
                    random.randint(self.n[0], self.n[1]),
                    activation=self.activation,
                )
            )
        model.add(keras.layers.Dense(1, activation="linear"))
        model.compile(loss=self.loss, optimizer="adam")
        return model

    def optimize_DNN(self, kfolds, trials, epochs, batch_size):
        """Performs DNN optimizing

        :param kfolds: number of folds for cross validation
        :type kfolds: int
        :param trials: number of trials
        :type trials: int
        :param epochs: number of epochs
        :type epochs: int
        :param batch_size: size of batch, defaults to 40
        :type batch_size: int, optional
        """
        self.epochs = epochs
        cv = KFold(kfolds)
        last = 0
        first = 1

        # Loop for different trials or models to train in order to find the
        for row in range(trials):

            # Create an auxiliar to clone in order to avoid undesired weight learning
            model_aux = self.create_random_network()
            train_score = []
            val_score = []

            # Loop that manage cross validation using training set
            for train_index, test_index in cv.split(self.X.values):
                # This sentence carefully clones the untrained model in each fold in order to avoid unwanted learning weights between them
                model = keras.models.clone_model(model_aux)
                model.compile(optimizer="adam", loss=self.loss)

                X_train, X_test = (
                    self.X.values[train_index],
                    self.X.values[test_index],
                )
                t_train, t_test = (
                    self.t.values[train_index],
                    self.t.values[test_index],
                )

                # Training of the model
                history = model.fit(
                    X_train,
                    t_train,
                    validation_data=(X_test, t_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                )

                train_score.append(history.history["loss"])
                val_score.append(history.history["val_loss"])

            # Mean validation score used for choose best DNN model
            score = np.mean(np.mean(val_score, axis=0))
            if score < last or first:
                first = 0
                last = np.mean(score)
                self.bestDNN = model
                self.scores = {
                    "val": val_score,
                    "train": train_score,
                }

    def get_best_hyperparameters(self):
        """Method to organize and get hyperparameters of the best DNN model"""

        model = self.bestDNN
        neurons = []
        activation = []
        nlayers = len(model.layers)

        for i in range(nlayers):
            neurons.append(model.layers[i].units)
            activation.append(model.layers[i].get_config()["activation"])

        lr = model.optimizer.learning_rate.value().numpy()
        self.params = {
            "Model": "DNN",
            "layers": int(nlayers),
            "neurons": str(neurons),
            "activation": str(activation),
            "optimizer": model.optimizer.get_config()["name"],
            "lr": lr,
        }
