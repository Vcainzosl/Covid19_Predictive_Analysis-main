from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class LR:
    def __init__(
        self,
        params_dist={
            "fit_intercept": ("True", "False"),
            "normalize": ("True", "False"),
        },
        **kwargs
    ):
        self.model = LinearRegression(**kwargs)
        self.model_name = "Linear Regression"
        self.params_dist = params_dist