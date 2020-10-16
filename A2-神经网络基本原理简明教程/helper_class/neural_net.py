import numpy as np
import math
from helper_class.training_history import TrainingHistory


# from matplotlib.colors import LogNorm
# from matplotlib import pyplot as plt


class NeuralNet(object):

    def __init__(self, params):
        self.params = params
        self.w = np.zeros((self.params.input_size, self.params.output_size))
        self.b = np.zeros((1, self.params.output_size))

    def __forward_batch(self, batch_x):
        matrix_z = np.dot(batch_x, self.w) + self.b
        return matrix_z

    @staticmethod
    def __backward_batch(batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dz = batch_z - batch_y
        dw = np.dot(batch_x.T, dz) / m
        db = dz.sum(axis=0, keepdims=True) / m
        return dw, db

    def __check_loss(self, data_reader):
        matrix_x, matrix_y = data_reader.get_whole_train_samples()
        m = matrix_x.shape[0]
        matrix_z = self.__forward_batch(matrix_x)
        loss = ((matrix_z - matrix_y) ** 2).sum() / m / 2
        return loss

    def __update(self, dw, db):
        self.w = self.w - self.params.eta * dw
        self.b = self.b - self.params.eta * db

    def inference(self, x):
        return self.__forward_batch(x)

    def train(self, data_reader, checkpoint=0.1):
        loss_history = TrainingHistory()
        loss = None
        epoch = 0
        iteration = 0
        if self.params.batch_size == -1:
            self.params.batch_size = data_reader.num_train
        max_iteration = math.ceil(data_reader.num_train / self.params.batch_size)
        checkpoint_iteration = int(max_iteration * checkpoint)

        for epoch in range(self.params.max_epoch):
            # print("epoch=%d" % epoch)
            data_reader.shuffle()
            for iteration in range(int(max_iteration)):
                batch_x, batch_y = data_reader.get_batch_train_samples(self.params.batch_size, iteration)
                batch_z = self.__forward_batch(batch_x)
                dw, db = self.__backward_batch(batch_x, batch_y, batch_z)
                self.__update(dw, db)

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    loss = self.__check_loss(data_reader)
                    loss_history.add_loss_history(total_iteration, loss)
                    # print(epoch, iteration, loss)
                    if loss < self.params.eps:
                        break
            if loss < self.params.eps:
                break
        loss_history.show_loss_history(self.params)
        print(self.w, self.b)
    #     self.loss_counter(data_reader, loss_history, self.params.batch_size, epoch * max_iteration + iteration)
    #
    # def loss_counter(self, data_reader, loss_history, batch_size, iteration):
    #     last_loss, result_w, result_b = loss_history.get_last()
    #     len1 = 50
    #     matrix_x, matrix_y = data_reader.get_whole_train_samples()
    #     w = np.linspace(result_w - 1, result_w + 1, len1)
    #     b = np.linspace(result_b - 1, result_b + 1, len1)
    #     matrix_w, matrix_b = np.meshgrid(w, b)
    #     m = matrix_x.shape[0]
    #     matrix_z = np.dot(matrix_x, matrix_w.ravel().reshape(1, len1 ** 2)) + matrix_b.ravel().reshape(1, len1 ** 2)
    #     loss1 = (matrix_z - matrix_y) ** 2
    #     loss2 = loss1.sum(axis=0, keepdims=True) / m
    #     loss3 = loss2.reshape(len1, len1)
    #     plt.contour(matrix_w, matrix_b, loss3, levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)
    #
    #     # show w,b trace
    #     w_history = loss_history.w_history
    #     b_history = loss_history.b_history
    #     plt.plot(w_history, b_history)
    #     plt.xlabel("w")
    #     plt.ylabel("b")
    #     title = str.format("batchsize={0}, iteration={1}, eta={2}, w={3:.3f}, b={4:.3f}",
    #                        batch_size, iteration, self.params.eta, result_w, result_b)
    #     plt.title(title)
    #
    #     plt.axis([result_w - 1, result_w + 1, result_b - 1, result_b + 1])
    #     plt.show()
