import numpy as np
import matplotlib as mtl
from matplotlib import pyplot as plt

mtl.use('TkAgg')

error = 1e-3


def draw_fun(X, Y):  # c = np.sqrt(np.log(x*x))
    x = np.linspace(1.1, 10)  # x范围 [1,10]
    _, _, c = forward(x)
    plt.plot(x, c)  # 画出c = np.sqrt(np.log(x*x))

    plt.plot(X, Y, 'x')

    d = 1 / (x * np.sqrt(np.log(x ** 2)))  # c = np.sqrt(np.log(x*x)) 的导数
    plt.plot(x, d)
    plt.show()


def forward(x):
    a = x * x
    b = np.log(a)
    c = np.sqrt(b)
    return a, b, c


def backward(x, a, b, c, y):
    loss = c - y
    delta_c = loss
    delta_b = delta_c * 2 * np.sqrt(b)
    delta_a = delta_b * a
    delta_x = delta_a / 2 / x
    return loss, delta_c, delta_b, delta_a, delta_x


def update(x, delta_x):
    x = x - delta_x
    if x < 1:
        x = 1.1
    return x


def back_propagate(x, y):
    X, Y = [], []
    while True:
        # forward
        print("forward...")
        a, b, c = forward(x)
        print("x=%f,a=%f,b=%f,c=%f" % (x, a, b, c))
        X.append(x)
        Y.append(c)
        # backward
        print("backward...")
        loss, delta_x, delta_a, delta_b, delta_c = backward(x, a, b, c, y)
        if abs(loss) < error:
            print("done!")
            break
        # update x
        x = update(x, delta_x)
        print("delta_c=%f, delta_b=%f, delta_a=%f, delta_x=%f\n" %
              (delta_c, delta_b, delta_a, delta_x))
    draw_fun(X, Y)


if __name__ == "__main__":
    back_propagate(1.5, 1.8)
