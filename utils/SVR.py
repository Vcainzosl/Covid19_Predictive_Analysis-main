from sklearn.svm import SVR as SVr
from scipy.stats import loguniform


class SVR:
    """Class to encapsulate SVR model methods"""

    def __init__(
        self,
        params_dist={
            "kernel": ("rbf", "poly", "sigmoid"),
            "C": loguniform(1e-5, 100),
        },
        **kwargs
    ):
        """Contructor of SVR class

        :param params_dist: hyperparameter range values, defaults to { "kernel": ("rbf", "poly", "sigmoid"), "C": loguniform(1e-5, 10), }
        :type params_dist: dict, optional
        """
        self.model = SVr(**kwargs)
        self.model_name = "SVR"
        self.params_dist = params_dist