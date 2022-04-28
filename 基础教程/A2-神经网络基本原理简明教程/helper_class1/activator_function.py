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


class BenIdentity(Activator):
    def forward(self, z):
        # (sqrt(z * z + 1) -1) / 2 + z
        p1 = np.multiply(z, z)
        p2 = np.sqrt(p1 + 1)
        a = (p2 - 1) / 2 + z
        return a

    def backward(self, z, a, delta):
        da = z / (2 * np.sqrt(z ** 2 + 1)) + 1
        dz = np.multiply(da, delta)
        return dz, da


class Elu(Activator):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        return np.array([x if x > 0 else self.alpha * (np.exp(x) - 1) for x in z])

    def backward(self, z, a, delta):
        da = np.array([1 if x > 0 else self.alpha * np.exp(x) for x in a])
        dz = np.multiply(delta, da)
        return dz, da


class LeakyRelu(Activator):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        return np.array([x if x > 0 else self.alpha * x for x in z])

    def backward(self, z, a, delta):
        da = np.array([1 if x > 0 else self.alpha for x in a])
        dz = np.multiply(delta, da)
        return dz, da


class SoftPlus(Activator):
    def forward(self, z):
        a = np.log(1 + np.exp(z))
        return a

    def backward(self, z, a, delta):
        p = np.exp(z)
        da = p / (1 + p)
        dz = np.multiply(delta, da)
        return dz, da


class Step(Activator):
    def __init__(self, threshold):
        self.threshold = threshold

    def forward(self, z):
        a = np.array([1 if x > self.threshold else 0 for x in z])
        return a

    def backward(self, z, a, delta):
        da = np.zeros(a.shape)
        dz = da
        return dz, da
