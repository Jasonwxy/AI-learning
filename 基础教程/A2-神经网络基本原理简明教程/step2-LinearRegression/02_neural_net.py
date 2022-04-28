import numpy as np
from matplotlib import pyplot as plt
from helper_class1.data_reader import DataReader

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch04.npz'


class NeuralNet(object):

    def __init__(self, eta):
        """
        类初始化
        :param eta: 学习率需要指定
        """
        self.eta = eta
        self.w = 0
        self.b = 0

    def __forward(self, x):
        """
        前向计算，私有方法不对外 返回 x·w + b
        :param x: 样本数据变量x
        :return: 前向计算结果z
        """
        z = self.w * x + self.b
        return z

    @staticmethod
    def __backward(x, y, z):
        """
        反向传播，比对输入值x根据前向计算得出的结果z和实际结果的差值和梯度下降公式得到的w和b的误差值
        :param x: 样本数据变量x
        :param y: 样本数据实际结果y
        :param z: 前向计算结果z
        :return: w和b的误差值
        """
        dz = z - y  # dz(delta_z) z的误差值
        dw = dz * x
        db = dz
        return dw, db

    def __update(self, dw, db):
        """
        根据w和b的误差与学习率，更新w和b的值
        :param dw: w的误差
        :param db: b的误差
        """
        self.w = self.w - self.eta * dw
        self.b = self.b - self.eta * db

    def train(self, data_reader):
        """
        根据训练数据，将所有训练数据使用，得到最后的w和b
        :param data_reader: 训练数据读取
        """
        for i in range(data_reader.num_train):
            x, y = data_reader.get_single_train_sample(i)
            z = self.__forward(x)
            dw, db = self.__backward(x, y, z)
            self.__update(dw, db)

    def inference(self, x):
        return self.__forward(x)


def show_result(net, data_reader):
    x_set, y_set = data_reader.get_whole_train_samples()
    plt.plot(x_set, y_set, 'b.')

    p_x = np.linspace(0, 1, 10)
    p_z = net.inference(p_x)
    plt.plot(p_x, p_z, "r")
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()


if __name__ == "__main__":
    eta1 = 0.1
    net1 = NeuralNet(eta1)
    data_reader1 = DataReader(file_name)
    data_reader1.read_data()
    net1.train(data_reader1)
    print("w=%f,b=%f" % (net1.w, net1.b))
    show_result(net1, data_reader1)
