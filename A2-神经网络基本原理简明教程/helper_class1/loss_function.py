import numpy as np
from helper_class1.enum_def import NetType


class LossFunction(object):

    def __init__(self, net_type=NetType.Fitting):
        self.net_type = net_type

    def check_loss(self, matrix_a, matrix_y):
        m = matrix_y.shape[0]
        loss = 0
        if self.net_type == NetType.Fitting:
            loss = self.mse(matrix_a, matrix_y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.ce2(matrix_a, matrix_y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.ce3(matrix_a, matrix_y, m)
        elif self.net_type == NetType.BinaryTanh:
            loss = self.ce2_tanh(matrix_a, matrix_y, m)
        return loss

    @staticmethod
    def mse(matrix_a, matrix_y, count):
        return ((matrix_a - matrix_y) ** 2).sum() / count / 2

    @staticmethod
    def ce2(matrix_a, matrix_y, count):
        # loss = -yln(a)-(1-y)ln(1-a)
        # loss = np.sum(-(np.multiply(matrix_y, np.log(matrix_a)) + np.multiply(1 - matrix_y, np.log(1 - matrix_a))))
        loss = np.sum(-matrix_y * np.log(matrix_a) - (1 - matrix_y) * np.log(1 - matrix_a))
        return loss / count

    @staticmethod
    def ce2_tanh(matrix_a, matrix_y, count):
        # loss = -(1-y)ln((1-a)/2)-(1+y)ln((1+a)/2)
        loss = np.sum(-(1 - matrix_y) * np.log((1 - matrix_a) / 2) - (1 + matrix_y) * np.log((1 + matrix_a) / 2))
        return loss / count

    @staticmethod
    def ce3(matrix_a, matrix_y, count):
        # loss = -y*ln(a)
        loss = np.sum(-matrix_y * np.log(matrix_a))
        return loss / count
