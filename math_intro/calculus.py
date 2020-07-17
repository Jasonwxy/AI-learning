import numpy as np
import matplotlib
from IPython.core.interactiveshell import InteractiveShell
from matplotlib import pyplot as plt

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
    x = np.array(range(0, 11))
    y = np.array([0, 10])

    plt.plot(x, f(x), color='g')
    plt.plot(y, f(y), color='m')

    draw()


def demo2():
    x = [*range(0, 5), *np.arange(4, 6, 0.1), *range(6, 11)]
    y = [f(i) for i in x]

    plt.plot(x, y, color='lightgrey', marker='o', markerfacecolor='green', markeredgecolor='green', markersize=2)
    plt.plot(5, f(5), color='red', marker='o', markersize=5)

    draw()


def demo3():
    pass


if __name__ == "__main__":
    print([*range(0,24),*np.arange(24, 25, 0.1),*range(26,101)])
