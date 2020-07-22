import numpy as np
import matplotlib
import math
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


def demo():
    v = np.array([2, 1])
    v_mag = np.linalg.norm(v)  # 向量v的标量
    print(v_mag)

    origin = [0], [0]
    plt.axis('equal')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.quiver(*origin, *v, scale=10, color='r')
    draw()


def demo1():
    v1 = np.array([2, 1])
    v2 = np.array([-3, 2])
    d = np.cross(v1, v2)
    print(d)


if __name__ == '__main__':
    demo1()
