from utils.LR import LR
from utils.SVR import SVR


class Models:
    """Class to organize object models created of scikit-learn library"""

    def __init__(
        self,
        LinearRegression: bool = True,
        KNN: bool = False,
        SVr: bool = True,
    ):
        """Contructor of Models class

        :param LinearRegression: includes LR model, defaults to True
        :type LinearRegression: bool, optional
        :param SVr: includes SVR model, defaults to True
        :type SVr: bool, optional
        """
        self.models = list()
        if LinearRegression:
            self.models.append(LR())
        if SVr:
            self.models.append(SVR())
