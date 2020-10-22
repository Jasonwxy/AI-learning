import numpy as np
from matplotlib import pyplot as plt
from helper_class.data_reader import DataReader
from helper_class.hyper_parameters import HyperParameters
from helper_class.visualizer import draw_two_category_points
from helper_class.enum_def import NetType
from helper_class.neural_net import NeuralNet


class LogicDateReader(DataReader):
    def __init__(self, data_file=None):
        super().__init__(data_file)

    def read_logic_and_data(self):
        x = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        y = np.array([0, 0, 0, 1]).reshape(4, 1)
        self.__set_data(x, y)

    def read_logic_or_data(self):
        x = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        y = np.array([0, 1, 1, 1]).reshape(4, 1)
        self.__set_data(x, y)

    def read_logic_not_data(self):
        x = np.array([0, 1]).reshape(2, 1)
        y = np.array([1, 0]).reshape(2, 1)
        self.__set_data(x, y)

    def read_logic_nor_data(self):
        x = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        y = np.array([1, 0, 0, 0]).reshape(4, 1)
        self.__set_data(x, y)

    def read_logic_nand_data(self):
        x = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        y = np.array([1, 1, 1, 0]).reshape(4, 1)
        self.__set_data(x, y)

    def __set_data(self, x, y):
        self.x_train = self.x_raw = x
        self.y_train = self.y_raw = y
        self.num_train = self.x_raw.shape[0]


def test(net, reader):
    x, y = reader.get_whole_train_samples()
    a = net.inference(x)
    diff = np.abs(a - y)
    result = np.where(diff < 1e-2, True, False)
    return result.sum() == x.shape[0]


def draw_split_line(net):
    if net.w.shape[0] == 2:
        b = -net.b[0, 0] / net.w[1, 0]
        w = -net.w[0, 0] / net.w[1, 0]
    else:
        w = net.w[0]
        b = net.b[0]
    x = np.array([-0.1, 1.1])
    y = w * x + b
    plt.plot(x, y)
    plt.show()


def draw_source_data(data_reader, title, show=False):
    x, y = data_reader.get_whole_train_samples()
    plt.figure(figsize=(5, 5))
    plt.grid()
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.title(title)
    if data_reader.x_train.shape[1] == 1:
        draw_two_category_points(x[:, 0], np.zeros_like(x[:, 0]), y[:, 0], show=show)
    else:
        draw_two_category_points(x[:, 0], x[:, 1], y[:, 0], show=show)


def train(reader, title):
    draw_source_data(reader, title, show=True)
    num_input = reader.x_train.shape[1]
    num_output = 1
    hp = HyperParameters(num_input, num_output, eta=0.5, max_epoch=100000, batch_size=1, eps=2e-3,
                         net_type=NetType.BinaryClassifier)
    net = NeuralNet(hp)
    net.train(reader, checkpoint=1)
    print(test(net, reader))
    draw_source_data(reader, title, show=False)
    draw_split_line(net)


if __name__ == '__main__':
    reader1 = LogicDateReader()

    reader1.read_logic_and_data()
    train(reader1, "logic and operator")

    reader1.read_logic_or_data()
    train(reader1, "logic or operator")

    reader1.read_logic_not_data()
    train(reader1, "logic not operator")

    reader1.read_logic_nor_data()
    train(reader1, "logic nor operator")

    reader1.read_logic_nand_data()
    train(reader1, "logic nand operator")
