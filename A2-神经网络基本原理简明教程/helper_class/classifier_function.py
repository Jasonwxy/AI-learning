import numpy as np


class Logistic(object):
    @staticmethod
    def forward(z):
        return 1.0 / (1.0 + np.exp(-z))


class Tanh(object):
    @staticmethod
    def forward(z):
        return 2.0 / (1.0 + np.exp(-2 * z)) - 1.0
