import numpy as np
from .layer import Layer


class ClassificationLayer(Layer):

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.x = inputs
        self.z = self.classifier.forward(inputs)
        return self.z

    def backward(self, delta_in, flag):
        return delta_in


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
