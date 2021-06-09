from sklearn.model_selection import RandomizedSearchCV


class Optimizing:
    """Class to perform model optimizing"""

    def __init__(self, X, t, model: object, cv, trials, **kwargs):
        """Contructor of Optimizing class

        :param X: training dataset
        :type X: pandas.DataFrame
        :param t: training labels
        :type t: pandas.Series
        :param model: model of scikit-learn
        :type model: object
        :param cv: number of folds
        :type cv: int
        :param trials: number of trials
        :type trials: int
        """
        self.model = model
        self.params_dist = self.model.params_dist
        # Call optimizing method
        self.optimize_model(X, t, cv=cv, n_iter=trials, **kwargs)
        # Get results
        self.best_model = self.get_best_model()
        self.best_params = self.get_best_params()
        self.best_results = self.get_best_results()
        self.best_index = self.random_search.best_index_

    def optimize_model(self, X, t, **kwargs):
        """Performs optimizing model in hyperparameters range values

        :param X: training dataset
        :type X: pandas.DataFrame
        :param t: training label
        :type t: pandas.Series
        """
        # Using Random search with hyperparameters range values implicit define in each model
        clf = RandomizedSearchCV(
            self.model.model, self.model.params_dist, **kwargs
        )
        self.random_search = clf.fit(X, t)

    def get_best_model(self):
        """Get best model

        :return: best model
        :rtype: object Estimator
        """
        return self.random_search.best_estimator_

    def get_best_params(self):
        """Get best hyperparameters

        :return: best hyperparameters
        :rtype: dict
        """
        return {
            "Model": self.model.model_name,
            **self.random_search.best_params_,
        }

    def get_best_results(self):
        """Get best results

        :return: best validation and training results
        :rtype: dict
        """
        return self.random_search.cv_results_
