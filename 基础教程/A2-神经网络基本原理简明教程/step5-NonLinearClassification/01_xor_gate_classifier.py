import numpy as np
from helper_class2.hyper_parameters2 import HyperParameters
from helper_class2.data_reader import DataReader
from helper_class2.enum_def import InitialMethod, NetType
from helper_class2.neural_net2 import NeuralNet


class XORDataReader(DataReader):
    def __init__(self):
        pass

    def read_data(self):
        self.x_train_raw = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        self.y_train_raw = np.array([0, 1, 1, 0]).reshape(4, 1)
        self.x_train = self.x_train_raw
        self.y_train = self.y_train_raw

        self.num_category = 1
        self.num_feature = self.x_train_raw.shape[1]
        self.num_train = self.x_train_raw.shape[0]
        self.num_test = self.num_train

        self.x_test_raw = self.x_train_raw
        self.y_test_raw = self.y_train_raw
        self.x_test = self.x_test_raw
        self.y_test = self.y_test_raw
        self.x_dev = self.x_train
        self.y_dev = self.y_train


def run_test(data_reader, net):
    print("testing...")
    x, y = data_reader.get_test_set()
    print(x, y)
    print(data_reader.x_test, data_reader.y_test)
    a2 = net.inference(x)
    print("a2=", a2)
    diff = np.abs(a2 - y)
    result = np.where(diff < 1e-2, True, False)
    return result.sum() == data_reader.num_test


if __name__ == '__main__':
    reader = XORDataReader()
    reader.read_data()

    n_input = reader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch, eps = 0.1, 1, 10000, 0.005

    hp = HyperParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier,
                         InitialMethod.Xavier)

    net1 = NeuralNet(hp, 'xor_221')
    net1.train(reader, 100, True)
    net1.show_training_history()

    print(run_test(reader, net1))
