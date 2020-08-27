import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

# MSE Mean Square Error 均方差  单样本：loss = (z-y)**2/2 多样本：J=((z1-y1)**2+...+(zm-ym)**2)/2m

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch03.npz'


def target_function(x, w, b):
    return w * x + b


def create_simple_data(w, b, n):
    if os.path.exists(file_name):
        data = np.load(file_name)
        x = data["data"]
        y = data["label"]
    else:
        x = np.linspace(0, 1, num=n)
        noise = np.random.uniform(-0.5, 0.5, size=n)
        y = target_function(x, w, b) + noise
        np.savez(file_name, data=x, label=y)
    return x, y


def cost_function(y, z, count):
    c = (z - y) ** 2
    return c.sum() / count / 2


def show_result(ax, x, y, a, loss, title):
    ax.scatter(x, y)
    ax.plot(x, a, 'r')
    titles = str.format("{0} Loss={1:01f}", title, loss)
    ax.set_title(titles)


# 显示只变化b时的loss变化

def calculate_cost_b(x, y, n, w, b):
    B = np.arange(b - 1, b + 1, 0.05)
    losses = []
    for i in range(len(B)):
        z = w * x + B[i]
        loss = cost_function(y, z, n)
        losses.append(loss)
    plt.title("Loss according to b")
    plt.xlabel("b")
    plt.ylabel("J")
    plt.plot(B, losses, 'x')
    plt.show()


# 显示只变化w时的loss变化

def calculate_cost_w(x, y, n, w, b):
    W = np.arange(w - 1, w + 1, 0.05)
    losses = []
    for i in range(len(W)):
        z = W[i] * x + b
        loss = cost_function(y, z, n)
        losses.append(loss)
    plt.title("Loss according to w")
    plt.xlabel("w")
    plt.ylabel("J")
    plt.title = "Loss according to w"
    plt.plot(W, losses, 'o')
    plt.show()


# 显示同时变化w,b时loss的变化情况

def calculate_cost_wb(x, y, n, w, b):
    B = np.arange(b - 10, b + 10, 0.1)
    W = np.arange(w - 10, w + 10, 0.1)
    losses = np.zeros((len(W), len(B)))
    for i in range(len(W)):
        for j in range(len(B)):
            z = W[i] * x + B[j]
            loss = cost_function(y, z, n)
            losses[i, j] = loss
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(W, B, losses)
    plt.show()


# 在一张图上分区域画出b的4种取值的loss的情况
def show_cost_for_4b(x, y, n, w, b):
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    a1 = w * x + b - 1
    loss1 = cost_function(y, a1, n)
    show_result(ax1, x, y, a1, loss1, '2x+%d' % (b - 1))
    a2 = w * x + b - 0.5
    loss2 = cost_function(y, a2, n)
    show_result(ax2, x, y, a2, loss2, '2x+%d' % (b - 0.5))
    a3 = w * x + b
    loss3 = cost_function(y, a3, n)
    show_result(ax3, x, y, a3, loss3, '2x+%d' % b)
    a4 = w * x + b + 0.5
    loss4 = cost_function(y, a4, n)
    show_result(ax4, x, y, a4, loss4, '2x+%d' % (b + 0.5))
    plt.show()


# 在一张图上显示b的4种取值的比较

def show_all_4b(x, y, w, b):
    plt.scatter(x, y)
    z1 = w * x + b - 1
    plt.plot(x, z1)

    z2 = w * x + b - 0.5
    plt.plot(x, z2)

    z3 = w * x + b
    plt.plot(x, z3)

    z4 = w * x + b + 0.5
    plt.plot(x, z4)
    plt.show()


# 画出3D示意图

def show_3d_surface(x, y, m, w, b):
    fig = plt.figure()
    ax = Axes3D(fig)

    X = x.reshape(m, 1)
    Y = y.reshape(m, 1)

    len1 = 50
    lens = len1 * len1

    W = np.linspace(w - 2, w + 2, len1)
    B = np.linspace(b - 2, b + 2, len1)
    W, B = np.meshgrid(W, B)

    m = X.shape[0]
    Z = np.dot(X, W.ravel().reshape(1, lens)) + B.ravel().reshape(1, lens)
    loss1 = (Z - Y) ** 2
    loss2 = loss1.sum(axis=0, keepdims=True) / m / 2
    loss3 = loss2.reshape(len1, len1)
    ax.plot_surface(W, B, loss3, norm=LogNorm(), cmap='rainbow')
    plt.show()


def test_2d(x, y, m, w, b):
    s = 200
    W = np.linspace(w - 2, w + 2, s)
    B = np.linspace(b - 2, b + 2, s)
    LOSS = np.zeros((s, s))
    for i in range(len(W)):
        for j in range(len(B)):
            z = W[i] * x + B[j]
            loss = cost_function(y, z, m)
            LOSS[i, j] = round(loss, 2)
    print(LOSS)
    print("please wait for 20 seconds...")
    while True:
        X = []
        Y = []
        is_first = True
        loss = 0
        for i in range(len(W)):
            for j in range(len(B)):
                if LOSS[i, j] != 0:
                    if is_first:
                        loss = LOSS[i, j]
                        X.append(W[i])
                        Y.append(B[j])
                        LOSS[i, j] = 0
                        is_first = False
                    elif (LOSS[i, j] == loss) or (abs(loss / LOSS[i, j] - 1) < 0.02):
                        X.append(W[i])
                        Y.append(B[j])
                        LOSS[i, j] = 0
        if is_first:
            break
        plt.plot(X, Y, '.')

    plt.xlabel("w")
    plt.ylabel("b")
    plt.show()


def draw_contour(x, y, m, w, b):
    X = x.reshape(m, 1)
    Y = y.reshape(m, 1)

    len1 = 50
    lens = len1 * len1

    W = np.linspace(w - 2, w + 2, len1)
    B = np.linspace(b - 2, b + 2, len1)
    W, B = np.meshgrid(W, B)

    m = X.shape[0]
    Z = np.dot(X, W.ravel().reshape(1, lens)) + B.ravel().reshape(1, lens)
    loss1 = (Z - Y) ** 2
    loss2 = loss1.sum(axis=0, keepdims=True) / m / 2
    loss3 = loss2.reshape(len1, len1)
    plt.contour(W, B, loss3, levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap='rainbow')
    plt.show()


if __name__ == "__main__":
    m = 50
    b = 3
    w = 2
    x, y = create_simple_data(w, b, m)
    # plt.scatter(x, y)
    # plt.show()
    #
    # show_cost_for_4b(x, y, m, w, b)
    # show_all_4b(x, y, w, b)
    #
    # calculate_cost_b(x, y, m, w, b)
    # calculate_cost_w(x, y, m, w, b)
    #
    # calculate_cost_wb(x, y, m, w, b)
    #
    # show_3d_surface(x, y, m, w, b)
    # test_2d(x, y, m, w, b)
    draw_contour(x, y, m, w, b)
