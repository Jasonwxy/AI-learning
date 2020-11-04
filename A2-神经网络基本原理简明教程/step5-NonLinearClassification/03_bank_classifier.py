from matplotlib import pyplot as plt
from helper_class2.data_reader import DataReader
from helper_class2.enum_def import NetType, InitialMethod
from helper_class2.neural_net import NeuralNet
from helper_class2.hyper_parameters import HyperParameters
from helper_class2.visualizer import draw_three_category_points, show_classification_result_25d

train_data_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch11.train.npz'
test_data_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch11.test.npz'

if __name__ == '__main__':
    reader = DataReader(train_data_file, test_data_file)
    reader.read_data()
    reader.normalize_y(NetType.MultipleClassifier, base=1)

    # plt.figure(figsize=(6, 6))
    # draw_three_category_points(reader.x_train_raw[:, 0], reader.x_train_raw[:, 1], reader.y_train,
    #                            title="Source Data", show=True)

    reader.normalize_x()
    reader.shuffle()
    reader.generate_validation_set()

    n_input = reader.num_feature
    n_hidden = 16
    n_output = reader.num_category
    eta, batch_size, max_epoch, eps = 0.1, 10, 10000, 0.1

    hp = HyperParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier,
                         InitialMethod.Xavier)
    net = NeuralNet(hp, "bank_2x3")

    # net.load_result()
    net.train(reader, 100, True)
    net.show_training_history()

    plt.figure(figsize=(6, 6))
    draw_three_category_points(reader.x_train[:, 0], reader.x_train[:, 1], reader.y_train, title=hp.to_string())
    show_classification_result_25d(net, 50, hp.to_string())
    plt.show()
