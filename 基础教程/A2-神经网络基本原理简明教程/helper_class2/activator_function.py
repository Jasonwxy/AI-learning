import numpy as np


class Activator(object):
    def forward(self, z):
        pass

    def backward(self, z, a, delta):
        pass


class Identity(Activator):
    def forward(self, z):
        return z

    def backward(self, z, a, delta):
        return delta, a


class Sigmoid(Activator):
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z, a, delta):
        da = np.multiply(a, 1 - a)
        dz = np.multiply(delta, da)
        return dz, da


class Tanh(Activator):
    def forward(self, z):
        return 2.0 / (1.0 + np.exp(-2 * z)) - 1

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return dz, da


class Relu(Activator):
    def forward(self, z):
        a = np.maximum(z, 0)
        return a

    def backward(self, z, a, delta):
        da = np.zeros(z.shape)
        da[z > 0] = 1
        dz = da * delta
        return dz, da
