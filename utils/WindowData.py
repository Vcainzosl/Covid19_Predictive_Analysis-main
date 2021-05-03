import numpy as np


class WindowData:
    """Customized class wrapper for window data"""

    def __init__(self, data: np.ndarray):
        """Constructor method of wrapper class

        :param data: window data
        :type data: numpy.ndarray
        """
        self._data = data

    def build_sample(self) -> np.ndarray:
        """Concatenates window data arrays

        :return: sample with time series data for each variable
        :rtype: numpy.ndarray
        """
        return np.concatenate(self._data)