import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    return x * x


def derivative_function(x):
    return 2 * x


def draw_function():
    x = np.linspace(-2, 2)
    y = target_function(x)
    plt.plot(x, y)


def draw_gd(X):
    Y = []
    for i in range(len(X)):
        Y.append(target_function(X[i]))
    plt.plot(X, Y)


if __name__ == "__main__":
    x = 1.5
    eta = 0.3
    error = 1e-5
    X = []
    X.append(x)
    y = target_function(x)
    while y > error:
        x = x - eta * derivative_function(x)
        X.append(x)
        y = target_function(x)
        print("x=", x, "y=", y)
    draw_function()
    draw_gd(X)
    plt.show()
