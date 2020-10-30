import numpy as np
from matplotlib import pyplot as plt
from helper_class2.data_reader import DataReader
from helper_class2.enum_def import NetType, InitialMethod
from helper_class2.neural_net import NeuralNet
from helper_class2.hyper_parameters import HyperParameters

train_data_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch09.train.npz'
test_data_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch09.test.npz'


def show_result(net, reader, title):
    x, y = reader.x_train, reader.y_train
    plt.plot(x[:, 0], y[:, 0], '.', c='b')
    tx = np.linspace(0, 1, 100).reshape(100, 1)
    ty = net.inference(tx)
    plt.plot(tx, ty, 'x', c='r')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    reader = DataReader(train_data_file, test_data_file)
    reader.read_data()
    reader.generate_validation_set()

    n_input = 1
    n_hidden = 3
    n_output = 1
    eta = 0.7
    batch_size = 10
    max_epoch = 10000
    eps = 0.0005

    hp = HyperParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting,
                         InitialMethod.Xavier)
    net = NeuralNet(hp, "complex_131")

    # net.load_result()
    net.train(reader, 50, True)
    net.show_training_history()
    show_result(net, reader, hp.to_string())
