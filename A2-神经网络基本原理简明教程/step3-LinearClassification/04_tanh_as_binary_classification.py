import numpy as np
from matplotlib import pyplot as plt
from helper_class1.neural_net import NeuralNet
from helper_class1.data_reader import DataReader
from helper_class1.visualizer import draw_two_category_points
from helper_class1.enum_def import NetType
from helper_class1.hyper_parameters import HyperParameters

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch06.npz'


class TanhDataReader(DataReader):
    def to_zero_one(self):
        y = np.zeros((self.num_train, 1))
        for i in range(self.num_train):
            y[i, 0] = 2 * self.y_train[i, 0] - 1
        self.y_train = y
        self.y_raw = y


def draw_source_data(data_reader, show=False):
    x, y = data_reader.get_whole_train_samples()
    plt.figure(figsize=(6, 6))
    draw_two_category_points(x[:, 0], x[:, 1], y[:, 0], show=show)


def draw_split_line(net):
    b12 = -net.b[0, 0] / net.w[1, 0]
    w12 = -net.w[0, 0] / net.w[1, 0]
    x = np.linspace(0, 1, 10)
    y = w12 * x + b12
    plt.plot(x, y)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.show()


def draw_predicate_data(net):
    x = np.array([0.58, 0.92, 0.62, 0.55, 0.39, 0.29]).reshape(3, 2)
    a = net.inference(x)
    draw_two_category_points(x[:, 0], x[:, 1], a[:, 0], show=False, is_predicate=True)


if __name__ == '__main__':
    reader = TanhDataReader(file_name)
    reader.read_data()
    reader.to_zero_one()

    hp = HyperParameters(2, 1, eta=0.1, max_epoch=10000, batch_size=10, eps=1e-3, net_type=NetType.BinaryTanh)
    net1 = NeuralNet(hp)
    net1.train(reader, checkpoint=10)

    draw_source_data(reader, show=False)
    draw_predicate_data(net1)
    draw_split_line(net1)
