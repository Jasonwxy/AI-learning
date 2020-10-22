import numpy as np
from helper_class.data_reader import DataReader
from helper_class.hyper_parameters import HyperParameters
from helper_class.enum_def import NetType
from helper_class.neural_net import NeuralNet

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch07.npz'


def inference(net, reader):
    xt_raw = np.array([5, 1, 7, 6, 5, 6, 2, 7]).reshape(4, 2)
    xt = reader.normalize_predicate_data(xt_raw)
    output = net.inference(xt)
    r = np.argmax(output, axis=1) + 1
    print("output=",output)
    print("r=", r)


if __name__ == '__main__':
    num_category = 3
    reader1 = DataReader(file_name)
    reader1.read_data()
    reader1.normalize_x()
    reader1.to_one_hot(num_category, base=1)

    hp = HyperParameters(2, 3, eta=0.1, max_epoch=100000, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    net1 = NeuralNet(hp)
    net1.train(reader1, checkpoint=1)

    inference(net1, reader1)
