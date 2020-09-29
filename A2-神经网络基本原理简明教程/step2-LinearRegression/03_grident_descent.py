import numpy as np
from helper_class.data_reader import DataReader
from helper_class.hyper_parameters import HyperParameters
from helper_class.neural_net import NeuralNet
from matplotlib import pyplot as plt

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch04.npz'

data_reader = DataReader(file_name)
data_reader.read_data()


def single_gd():
    train(0.1, 100, 1, 0.02)


def mini_batch_gd():
    train(0.3, 100, 10, 0.02)


def full_batch_gd():
    train(0.5, 1000, -1, 0.02)


def train(eta, max_epoch, batch_size, eps):
    hp = HyperParameters(1, 1, eta=eta, max_epoch=max_epoch, batch_size=batch_size, eps=eps)
    net = NeuralNet(hp)
    net.train(data_reader)
    show_result(net)


def show_result(net):
    matrix_x, matrix_y = data_reader.get_whole_train_samples()
    # draw sample data
    plt.plot(matrix_x, matrix_y, "b.")
    # draw predication data
    px = np.linspace(0, 1, 5).reshape(5, 1)
    pz = net.inference(px)
    plt.plot(px, pz, "r")
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()


if __name__ == "__main__":
    single_gd()
    mini_batch_gd()
    full_batch_gd()
