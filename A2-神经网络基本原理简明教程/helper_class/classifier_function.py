import numpy as np


class Logistic(object):
    @staticmethod
    def forward(z):
        return 1.0 / (1.0 + np.exp(-z))
