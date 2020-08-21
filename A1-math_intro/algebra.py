import numpy as np
import matplotlib as mtl
from IPython.core.interactiveshell import InteractiveShell
from matplotlib import pyplot as plt
import math

mtl.use('TkAgg')
InteractiveShell.ast_node_interactivity = 'last_expr'


def draw(x_lab='x', y_lab='y'):
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.grid()
    plt.axhline()
    plt.axvline()
    plt.show()


def demo1():  # 线性方程

    x0 = 4 / 3
    y0 = -4 / 2
    m = 1.5
    x = np.array(range(-10, 11))  # 从 -10 到10取21个数据点
    # y = (3 * x - 4) / 2  # 对于函数值
    y = m * x + y0  # 用斜率截距形式表示函数

    plt.plot(x, y, color='blue')

    mx = [0, x0]
    my = [y0, y0 + m * x0]
    # plt.annotate('x', (x0, 0))
    plt.annotate('y', (0, y0))
    plt.plot(mx, my, color='red')
    draw()


def demo2():  # 线性方程组
    l1p1 = [16, 0]
    l1p2 = [0, 16]
    l2p1 = [25, 0]
    l2p2 = [0, 10]

    plt.plot(l1p1, l1p2, color='red')
    plt.plot(l2p1, l2p2, color='blue')
    draw()


def demo3():  # 指数，根和对数
    # 指数
    x = 5 ** 4
    print(x)

    # 根
    x = math.sqrt(9)  # 二次方根
    print(x)

    cr = math.pow(64, 1 / 6)  # 三次方根
    print(cr)

    # 对数
    x = math.log(16, 4)  # 以4为底
    print(x)

    print(math.log10(64))  # 以10为底
    print(math.log(64))  # 以e为底 自然对数


def demo4():  # 幂运算
    x = np.array(range(-10, 11))
    y = 3 * x ** 3

    plt.plot(x, y, color='red')
    draw()


def demo5():  # 指数增长
    x = np.array(range(-10, 11))
    y = 2.0 ** x
    plt.plot(x, y, color='red')
    draw()


def demo6():  # 二次方程
    x = np.array(range(-9, 9))
    y = 2 * x ** 2 + 2 * x - 4
    plt.plot(x, y, color='red')
    draw()


def plot_parabola(a, b, c):  # y = ax*x + bx +c
    vx = (-1 * b) / (2 * a)  # 顶点x的值  顶点处导数为0, 2ax+b=0  x =-b/2a
    vy = a * vx ** 2 + b * vx + c

    minx = int(vx - 10)
    maxx = int(vx + 11)
    x = np.array(range(minx, maxx))
    y = a * x ** 2 + b * x + c
    miny = y.min()
    maxy = y.max()

    plt.plot(x, y, color='blue')
    sx = [vx, vx]
    sy = [miny, maxy]
    plt.plot(sx, sy, color='red')
    plt.scatter(vx, vy, color='black')

    draw()


def plot_parabola_from_formula(a, b, c):  # 画出y = ax*x + bx +c 的抛物线和抛物线的解

    if b ** 2 - 4 * a * c > 0:
        x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

        plt.scatter([x1, x2], [0, 0], color='green')
        plt.annotate('x1', (x1, 0))
        plt.annotate('x2', (x2, 0))

    plot_parabola(a, b, c)


def f(x):
    return x ** 2 + 2


def plot_formula():
    x = np.array(range(-100, 101))
    plt.plot(x, f(x), color='red')
    draw('x', 'f(x)')


def demo7():
    def g(x1):
        if x1 != 0:
            return (12 / (2 * x1)) ** 2

    x = range(-100, 101)
    y = [g(a) for a in x]

    plt.plot(x, y, color='red')
    draw('x', 'g(x)')


def demo8():
    def g(x1):
        if x1 >= 0:
            return 2 * np.sqrt(x1)

    x = range(0, 101)
    y = [g(a) for a in x]

    plt.plot(x, y, color='red')
    plt.plot(0, g(0), color='black', marker='o', markerfacecolor='black', markersize=8)
    draw()


def demo9():
    def g(x1):
        if 0 <= x1 <= 5:
            return x1 + 2

    x = range(0, 101)
    y = [g(a) for a in x]

    plt.plot(x, y, color='red')
    plt.plot(0, g(0), color='black', marker='o', markerfacecolor='black', markersize=8)
    plt.plot(5, g(5), color='black', marker='o', markerfacecolor='black', markersize=8)
    draw()


if __name__ == "__main__":
    demo9()
