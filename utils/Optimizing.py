from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utils.Splitter import Splitter


class Optimizing:
    def __init__(
        self,
        X,
        t,
        model: object,
        filter_rate=10,
        **kwargs,
    ):
        self.model = model
        self.params_dist = self.model.params_dist
        self.optimize_model(X, t, filter_rate=10, **kwargs)
        self.best_model = self.get_best_model()
        self.best_params = self.get_best_params()
        self.best_results = self.get_best_results()

    def optimize_model(self, X, t, filter_rate=10, **kwargs):
        clf = RandomizedSearchCV(
            self.model.model,
            self.model.params_dist,
            **Splitter(RandomizedSearchCV, **kwargs).kwargs,
        )
        self.random_search = clf.fit(X, t, **Splitter(clf.fit, **kwargs).kwargs)
        estimator = self.random_search.best_estimator_
        param_grid = {}
        self.filter = False
        for i in self.random_search.best_params_:
            if isinstance(self.random_search.best_params_[i], (int, float)):
                self.filter = True
                param_grid[i] = np.arange(
                    self.random_search.best_params_[i] / filter_rate,
                    self.random_search.best_params_[i] * filter_rate,
                )
        if self.filter:
            clf = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                **kwargs,
            )
            self.filter_search = clf.fit(
                X, t, **Splitter(clf.fit, **kwargs).kwargs
            )
            self.params_dist = param_grid

    def get_best_model(self):
        if self.filter:
            return self.filter_search.best_estimator_
        else:
            return self.random_search.best_estimator_

    def get_best_params(self):
        if self.filter:
            return {
                "Model": self.model.model_name,
                **self.filter_search.best_params_,
            }
        else:
            return {
                "Model": self.model.model_name,
                **self.random_search.best_params_,
            }

    def get_best_results(self):
        if self.filter:
            return self.filter_search.cv_results_
        else:
            return self.random_search.cv_results_
