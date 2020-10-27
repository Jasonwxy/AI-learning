import numpy as np
from matplotlib import pyplot as plt


def draw_two_category_points(x1, x2, y, xlabel='x1', ylabel='x2', title=None, show=False, is_predicate=False):
    colors = ['b', 'r']
    shapes = ['s', 'x']
    assert (x1.shape[0] == x2.shape[0] == y.shape[0])
    count = x1.shape[0]
    for i in range(count):
        j = int(round(y[i]))
        if j < 0:
            j = 0
        if is_predicate:
            plt.scatter(x1[i], x2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(x1[i], x2[i], color=colors[j], marker=shapes[j], zorder=10)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def draw_three_category_points(x1, x2, y, xlabel='x1', ylabel='x2', title=None, show=False, is_predicate=False):
    colors = ['b', 'r', 'g']
    shapes = ['s', 'x', 'o']
    assert (x1.shape[0] == x2.shape[0] == y.shape[0])
    count = x1.shape[0]
    for i in range(count):
        j = int(np.argmax(y[i]))
        if is_predicate:
            plt.scatter(x1[i], x2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(x1[i], x2[i], color=colors[j], marker=shapes[j], zorder=10)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
