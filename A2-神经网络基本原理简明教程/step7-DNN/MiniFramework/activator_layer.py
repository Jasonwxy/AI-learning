import numpy as np
from .layer import Layer

class ActivatorLayer(Layer):

    def __init__(self, activator):
        super().__init__()
        self.activator = activator

    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.x = inputs
        self.z =  self.activator.forward(inputs)
        return self.z

    def backward(self, delta_in, flag):
        return self.activator.backward(self.x,self.z,delta_in)


class Activator(object):
    def forward(self, z):
        pass

    def backward(self, z, a, delta):
        pass


class Identity(Activator):
    def forward(self, z):
        return z

    def backward(self, z, a, delta):
        return delta


class Sigmoid(Activator):
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z, a, delta):
        da = np.multiply(a, 1 - a)
        dz = np.multiply(delta, da)
        return dz


class Tanh(Activator):
    def forward(self, z):
        return 2.0 / (1.0 + np.exp(-2 * z)) - 1.0

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return dz


class Relu(Activator):
    def forward(self, z):
        a = np.maximum(z, 0)
        return a

    def backward(self, z, a, delta):
        da = np.zeros(z.shape)
        da[z > 0] = 1
        dz = da * delta
        return dz
