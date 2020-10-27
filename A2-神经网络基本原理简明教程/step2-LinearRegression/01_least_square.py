from helper_class1.data_reader import DataReader

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch04.npz'


def method1(X, Y, m):
    x_mean = X.mean()
    p = sum(Y * (X - x_mean))
    q = sum(X * X) - sum(X) * sum(X) / m
    return p / q


def method2(X, Y, m):
    p = sum(X * (Y - Y.mean()))
    q = sum(X * X) - X.mean() * sum(X)
    return p / q


def method3(X, Y, m):
    p = m * sum(X * Y) - sum(X) * sum(Y)
    q = m * sum(X * X) - sum(X) * sum(X)
    return p / q


def calculate_b_1(X, Y, w, m):
    return sum(Y - w * X) / m


def calculate_b_2(X, Y, w):
    return Y.mean() - w * X.mean()


if __name__ == "__main__":
    reader = DataReader(file_name)
    reader.read_data()
    X, Y = reader.get_whole_train_samples()
    m = X.shape[0]
    w1 = method1(X, Y, m)
    b1 = calculate_b_1(X, Y, w1, m)

    w2 = method2(X, Y, m)
    b2 = calculate_b_2(X, Y, w2)

    w3 = method3(X, Y, m)
    b3 = calculate_b_1(X, Y, w3, m)

    print("w1=%f, b1=%f" % (w1, b1))
    print("w2=%f, b2=%f" % (w2, b2))
    print("w3=%f, b3=%f" % (w3, b3))

    # 梯度下降
    eta = 0.1
    w, b = 0.0, 0.0
    for i in range(reader.num_train):
        # get x and y value for one sample
        xi = X[i]
        yi = Y[i]
        # 公式1
        zi = xi * w + b
        # 公式3
        dz = zi - yi
        # 公式4
        dw = dz * xi
        # 公式5
        db = dz
        # update w,b
        w = w - eta * dw
        b = b - eta * db

    print("w=", w)
    print("b=", b)
