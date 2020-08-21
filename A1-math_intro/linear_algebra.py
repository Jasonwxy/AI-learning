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
    v1 = np.array([2, 1, -1])
    v2 = np.array([-2, 3, 2])
    d = np.dot(v1, v2)  # 向量点乘
    print(d)
    c = np.cross(v1, v2)  # 向量叉乘
    print(c)


def matrix_demo():
    m1 = np.matrix([[1, 3, 2], [2, 3, 3], [0, 1, -1]])
    m2 = np.matrix([[2, 1, 4], [2, 5, 1], [2, -2, 3]])
    e = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    print(np.add(m1, m2))  # add 求和
    print(np.subtract(m1, m2))  # subtract 求差
    print(np.negative(m1))  # negative 求负
    print(np.transpose(m1))  # transpose 矩阵转换
    print(m1.T)  # 矩阵转换  同 transpose
    print(2 * m1)  # 矩阵乘以标量
    print(np.dot(m1, m2.T))  # dot 矩阵相乘
    print(m1 @ m2.T)  # @号 矩阵相乘 同上
    print(m1 @ e)  # 矩阵乘以单位矩阵 还是矩阵本身
    print(np.linalg.inv(m1))  # 逆矩阵 A·B=I(单位矩阵) AB互为 逆矩阵
    print(m1.I)  # 取逆矩阵  同上


def demo2():  # 解方程组 3x+5y=39   4x+2y=10
    a = np.matrix([[3, 5], [4, 2]])
    b = np.matrix([[39], [10]])
    print(np.linalg.solve(a, b))  # solve方法


def demo3():
    a = np.matrix([[2, 3], [5, 2]])
    v = np.array([1, 2])
    print(a @ v)  # 矩阵A和向量v T(v) = Av  称之为向量v的变换T
    a1 = np.matrix([[2, 0], [0, 2]])
    print(a1 @ v)  # 2倍放大
    a2 = np.matrix([[0, 1], [-1, 0]])
    print(a2 @ v)  # 顺时针转90°


def demo4():
    a = np.matrix([[3, 1], [1, 3]])
    l, q = np.linalg.eig(a)  # 求特征值和特征向量
    lv = np.diag(l)
    q_inv = np.linalg.inv(q)
    print(q)
    print(lv)
    print(q_inv)
    # v = np.array([1,3])
    print(np.around(q @ lv @ q_inv))
    print(np.around(a))


if __name__ == '__main__':
    demo4()
