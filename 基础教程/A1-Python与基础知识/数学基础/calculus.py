import numpy as np
import matplotlib
from IPython.core.interactiveshell import InteractiveShell
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy import integrate

matplotlib.use('TkAgg')
InteractiveShell.ast_node_interactivity = 'last_expr'


def draw(x_lab='x', y_lab='y'):
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.grid()
    plt.axhline()
    plt.axvline()
    plt.show()


def f(x):
    return x ** 2 + x


def demo1():
    x = np.array(range(11))
    y = np.array([0, 10])

    plt.plot(x, f(x), color='g')
    plt.plot(y, f(y), color='m')

    draw()


def demo2():
    x = [*range(5), *np.arange(4, 6, 0.1), *range(6, 11)]
    y = [f(i) for i in x]

    plt.plot(x, y, color='lightgrey', marker='o', markerfacecolor='green', markeredgecolor='green', markersize=2)
    plt.plot(5, f(5), color='red', marker='o', markersize=5)

    draw()


def g(x):
    return 3 * x ** 2 + 2 * x + 1


def demo3():
    x = range(11)
    y = [g(a) for a in x]

    fig, ax = plt.subplots()
    plt.plot(x, y, color='red')
    ix = np.linspace(0, 3)
    iy = g(ix)
    verts = [(0, 0)] + list(zip(ix, iy)) + [(3, 0)]
    poly = Polygon(verts, facecolor='orange')
    ax.add_patch(poly)

    draw()


if __name__ == "__main__":
    # demo3()
    res = integrate.quad(lambda x: g(x), 0, 3)
    print(res)
