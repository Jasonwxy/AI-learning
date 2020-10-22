import numpy as np


class Logistic(object):
    @staticmethod
    def forward(z):
        return 1.0 / (1.0 + np.exp(-z))


class Tanh(object):
    @staticmethod
    def forward(z):
        return 2.0 / (1.0 + np.exp(-2 * z)) - 1.0


class SoftMax(object):
    @staticmethod
    def forward(z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a
