import numpy as np
from matplotlib import pyplot as plt
from helper_class2.enum_def import NetType


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


def show_classification_result_25d(net, count, title):
    x = np.linspace(0, 1, count)
    y = np.linspace(0, 1, count)
    matrix_x, matrix_y = np.meshgrid(x, y)
    z = np.zeros((count, count))
    input_val = np.hstack((matrix_x.ravel().reshape(count * count, 1), matrix_y.ravel().reshape(count * count, 1)))
    output = net.inference(input_val)
    if net.hp.net_type == NetType.BinaryClassifier:
        z = output.reshape(count, count)
    elif net.hp.net_type == NetType.MultipleClassifier:
        sm = np.argmax(output, axis=1)
        z = sm.reshape(count, count)

    plt.contourf(matrix_x, matrix_y, z, cmap=plt.cm.Spectral, zorder=1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
