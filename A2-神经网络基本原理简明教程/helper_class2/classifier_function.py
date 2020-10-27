import numpy as np


class Classifier(object):
    def forward(self, z):
        pass


class Logistic(Classifier):
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))


class SoftMax(Classifier):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a
