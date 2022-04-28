import numpy as np
from matplotlib import pyplot as plt
from helper_class1.hyper_parameters import HyperParameters
from helper_class1.neural_net import NeuralNet
from helper_class1.enum_def import NetType
from helper_class1.data_reader import DataReader

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch09.train.npz'


class DataReadEx(DataReader):
    def add(self, num):
        for i in range(num - 1):
            x = self.x_train[:, 0:1] ** (i + 2)
            self.x_train = np.hstack((self.x_train, x))


def show_result(net, reader, title, num):
    x, y = reader.x_train, reader.y_train
    plt.plot(x[:, 0], y[:, 0], '.', c='b')
    tx = np.linspace(0, 1, 100).reshape(100, 1)

    for i in range(num - 1):
        tx = np.hstack((tx, tx[:, 0:1] ** (i + 2)))

    ty = net.inference(tx)
    plt.plot(tx[:, 0:1], ty, 'x', c='r')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    num_input = 10
    data_reader = DataReadEx(file_name)
    data_reader.read_data()
    data_reader.add(num_input)
    print(data_reader.x_train.shape)

    hp = HyperParameters(num_input, 1, eta=0.2, max_epoch=50000, batch_size=10, eps=1e-3, net_type=NetType.Fitting)
    net1 = NeuralNet(hp)
    net1.train(data_reader, checkpoint=10)
    show_result(net1, data_reader, hp.to_string(), num_input)
