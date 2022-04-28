import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def target_function(x, y):
    return x ** 2 + np.sin(y) ** 2


def derivative_function(theta):
    x = theta[0]
    y = theta[1]
    return np.array([2 * x, 2 * np.sin(y) * np.cos(y)])


def show_3d_surface(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)

    v = np.linspace(-3, 3, 100)
    len_v = len(v)
    print(len_v)
    X, Y = np.meshgrid(v, v)
    R = np.zeros((len_v, len_v))
    for i in range(len_v):
        for j in range(len_v):
            R[i, j] = target_function(X[i, j], Y[i, j])
    ax.plot_surface(X, Y, R, cmap='rainbow')
    plt.plot(x, y, z, color="red",linewidth=3)
    plt.show()


if __name__ == "__main__":
    theta = np.array([3, 1])
    eta = 0.1
    error = 1e-2
    X = []
    Y = []
    Z = []
    while True:
        print(theta)
        x = theta[0]
        y = theta[1]
        z = target_function(x, y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        print("x=%f, y=%f, z=%f" % (x, y, z))
        d_theta = derivative_function(theta)
        print('   ', theta)
        theta = theta - eta * d_theta
        if z < error:
            break
    show_3d_surface(X, Y, Z)
