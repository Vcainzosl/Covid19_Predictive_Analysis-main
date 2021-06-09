from sklearn.linear_model import LinearRegression


class LR:
    """Class to encapsulate Linear Regression model methods"""

    def __init__(
        self,
        params_dist={
            "fit_intercept": ("True", "False"),
            "normalize": ("True", "False"),
        },
        **kwargs
    ):
        """Contructor of LR class

        :param params_dist: hyperparameters range values, defaults to { "fit_intercept": ("True", "False"), "normalize": ("True", "False"), }
        :type params_dist: dict, optional
        """
        self.model = LinearRegression(**kwargs)
        self.model_name = "Linear Regression"
        self.params_dist = params_dist