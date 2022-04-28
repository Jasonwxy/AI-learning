import numpy as np
import matplotlib.pyplot as plt

from helper_class1.data_reader import DataReader
from helper_class1.neural_net import NeuralNet
from helper_class1.hyper_parameters import HyperParameters
from helper_class1.enum_def import NetType
from helper_class1.visualizer import draw_two_category_points

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch06.npz'


def draw_split_line(net):
    b12 = -net.b[0, 0] / net.w[1, 0]
    w12 = -net.w[0, 0] / net.w[1, 0]
    x = np.linspace(0, 1, 10)
    y = w12 * x + b12
    plt.plot(x, y)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.show()


def draw_source_data(data_reader, show=False):
    x, y = data_reader.get_whole_train_samples()
    plt.figure(figsize=(6, 6))
    draw_two_category_points(x[:, 0], x[:, 1], y[:, 0], show=show)


def draw_predicate_data(net):
    x = np.array([0.58, 0.92, 0.62, 0.55, 0.39, 0.29]).reshape(3, 2)
    a = net.inference(x)
    draw_two_category_points(x[:, 0], x[:, 1], a[:, 0], show=False, is_predicate=True)


if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.read_data()
    draw_source_data(reader, show=True)

    hp = HyperParameters(2, 1, eta=0.1, max_epoch=100000, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    net1 = NeuralNet(hp)
    net1.train(reader, checkpoint=10)

    draw_source_data(reader, show=False)
    draw_predicate_data(net1)
    draw_split_line(net1)
