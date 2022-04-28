import numpy as np
from helper_class1.data_reader import DataReader
from helper_class1.hyper_parameters import HyperParameters
from helper_class1.neural_net import NeuralNet

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch05.npz'

if __name__ == "__main__":
    reader = DataReader(file_name)
    reader.read_data()
    reader.normalize_x()
    reader.normalize_y()

    hp = HyperParameters(2, 1, eta=0.01, max_epoch=200, batch_size=10, eps=1e-5)
    net = NeuralNet(hp)
    net.train(reader, checkpoint=0.1)
    x1 = 15
    x2 = 93
    x = np.array([x1, x2]).reshape(1, 2)
    x_new = reader.normalize_predicate_data(x)
    z = net.inference(x_new)
    print('z=', z)
    z_true = z * reader.y_norm[0, 1] + reader.y_norm[0, 0]
    print('z_true=', z_true)
