import numpy as np
from matplotlib import pyplot as plt
from helper_class1.data_reader import DataReader
from helper_class1.hyper_parameters import HyperParameters
from helper_class1.enum_def import NetType
from helper_class1.neural_net import NeuralNet
from helper_class1.visualizer import draw_three_category_points

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch07.npz'


def draw_source_data(x, y, show=False):
    plt.figure(figsize=(6, 6))
    draw_three_category_points(x[:, 0], x[:, 1], y[:], xlabel="x1", ylabel="x2", show=show)


def draw_result(net, x, y):
    b12 = (net.b[0, 1] - net.b[0, 0]) / (net.w[1, 0] - net.w[1, 1])
    w12 = (net.w[0, 1] - net.w[0, 0]) / (net.w[1, 0] - net.w[1, 1])

    b23 = (net.b[0, 2] - net.b[0, 1]) / (net.w[1, 1] - net.w[1, 2])
    w23 = (net.w[0, 2] - net.w[0, 1]) / (net.w[1, 1] - net.w[1, 2])

    b13 = (net.b[0, 2] - net.b[0, 0]) / (net.w[1, 0] - net.w[1, 2])
    w13 = (net.w[0, 2] - net.w[0, 0]) / (net.w[1, 0] - net.w[1, 2])

    p13, = draw_split_line(w13, b13, 'r')
    p23, = draw_split_line(w23, b23, 'b')
    p12, = draw_split_line(w12, b12, 'g')

    plt.legend([p13, p23, p12], ["13", "23", "12"])
    plt.axis([-0.1, 1.1, -0.1, 1.1])

    draw_three_category_points(x[:, 0], x[:, 1], y[:], xlabel="x1", ylabel="x2", show=True, is_predicate=True)


def draw_split_line(w, b, color):
    x = np.linspace(0, 1, 2)
    y = w * x + b
    return plt.plot(x, y, c=color)


if __name__ == '__main__':
    num_category = 3
    reader = DataReader(file_name)
    reader.read_data()
    reader.to_one_hot(num_category, base=1)
    draw_source_data(reader.x_raw, reader.y_train, show=True)
    reader.normalize_x()

    hp = HyperParameters(2, 3, eta=0.1, max_epoch=40000, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    net1 = NeuralNet(hp)
    net1.train(reader, checkpoint=1)

    xt_raw = np.array([5, 1, 7, 6, 5, 6, 2, 7]).reshape(4, 2)
    xt = reader.normalize_predicate_data(xt_raw)
    output = net1.inference(xt)
    r = np.argmax(output, axis=1) + 1
    print("output=", output)
    print("r=", r)

    draw_source_data(reader.x_train, reader.y_train)
    draw_result(net1, xt, output)
