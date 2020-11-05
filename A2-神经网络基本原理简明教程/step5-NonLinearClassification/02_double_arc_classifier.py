from helper_class2.data_reader import DataReader
from helper_class2.enum_def import NetType, InitialMethod
from helper_class2.neural_net2 import NeuralNet
from helper_class2.hyper_parameters2 import HyperParameters

train_data_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch10.train.npz'
test_data_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch10.test.npz'

if __name__ == '__main__':
    reader = DataReader(train_data_file, test_data_file)
    reader.read_data()
    reader.normalize_x()
    reader.shuffle()
    reader.generate_validation_set()

    n_input = reader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch, eps = 0.1, 5, 10000, 0.08

    hp = HyperParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier,
                         InitialMethod.Xavier)
    net = NeuralNet(hp, "arc_221")
    net.train(reader, 5, True)
    net.show_training_history()
