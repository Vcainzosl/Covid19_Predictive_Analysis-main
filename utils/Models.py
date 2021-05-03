from utils.LR import LR


class Models:
    def __init__(
        self,
        LinearRegression: bool = True,
        LDA: bool = False,
        KNN: bool = False,
        SVC: bool = False,
    ):
        self.models = list()
        if LinearRegression:
            self.models.append(LR())
        if LDA:
            pass
        if KNN:
            pass
        if SVC:
            pass
