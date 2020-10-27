import numpy as np
from matplotlib import pyplot as plt
from helper_class.activator_function import Identity, Sigmoid, Tanh, Step, SoftPlus, Elu, Relu, LeakyRelu, BenIdentity


def draw(start, end, func, label1, label2):
    z = np.linspace(start, end, 200)
    a = func.forward(z)
    dz, da = func.backward(z, a, 1)

    p1, = plt.plot(z, a)
    p2, = plt.plot(z, da)
    plt.legend([p1, p2], [label1, label2])
    plt.grid()

    plt.xlabel('input: z')
    plt.ylabel('output: a')
    plt.title(label1)
    plt.show()


if __name__ == '__main__':
    draw(-7, 7, Identity(), "Identity Function", "Derivative of Identity")
    draw(-7, 7, Sigmoid(), "Sigmoid Function", "Derivative of Sigmoid")
    draw(-7, 7, Tanh(), "Tanh Function", "Derivative of Tanh")
    draw(-5, 5, Step(0.3), "Step Function", "Derivative of Step")
    draw(-5, 5, Relu(), "Relu Function", "Derivative of Relu")
    draw(-4, 4, Elu(0.8), "ELU Function", "Derivative of ELU")
    draw(-5, 5, LeakyRelu(0.01), "Leaky Relu Function", "Derivative of Leaky Relu")
    draw(-5, 5, SoftPlus(), "SoftPlus Function", "Derivative of SoftPlus")
    draw(-7, 7, BenIdentity(), "BenIdentity Function", "Derivative of BenIdentity")
