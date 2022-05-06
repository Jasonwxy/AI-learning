import numpy as np
from helper_class1.data_reader import DataReader
from helper_class1.neural_net import NeuralNet
from helper_class1.hyper_parameters import HyperParameters
from helper_class1.enum_def import NetType

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch06.npz'

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.read_data()
    hp = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    net = NeuralNet(hp)
    net.train(reader, checkpoint=1)

    x_predicate = np.array([0.58, 0.92, 0.62, 0.55, 0.39, 0.29]).reshape(3, 2)
    a = net.inference(x_predicate)
    print(a)
