import numpy as np
import math
from helper_class1.training_history import TrainingHistory
from helper_class1.enum_def import NetType
from helper_class1.classifier_function import Logistic, Tanh, SoftMax
from helper_class1.loss_function import LossFunction


class NeuralNet(object):

    def __init__(self, params):
        self.params = params
        self.w = np.zeros((self.params.input_size, self.params.output_size))
        self.b = np.zeros((1, self.params.output_size))

    def __forward_batch(self, batch_x):
        matrix_z = np.dot(batch_x, self.w) + self.b
        if self.params.net_type == NetType.BinaryClassifier:
            matrix_z = Logistic.forward(matrix_z)
        elif self.params.net_type == NetType.BinaryTanh:
            matrix_z = Tanh.forward(matrix_z)
        elif self.params.net_type == NetType.MultipleClassifier:
            matrix_z = SoftMax.forward(matrix_z)
        return matrix_z

    def __backward_batch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dz = batch_z - batch_y
        if self.params.net_type == NetType.BinaryTanh:
            dz = 2 * dz
        dw = np.dot(batch_x.T, dz) / m
        db = dz.sum(axis=0, keepdims=True) / m
        return dw, db

    def __check_loss(self, loss_fun, data_reader):
        matrix_x, matrix_y = data_reader.get_whole_train_samples()
        # m = matrix_x.shape[0]
        matrix_z = self.__forward_batch(matrix_x)
        # loss = ((matrix_z - matrix_y) ** 2).sum() / m / 2
        return loss_fun.check_loss(matrix_z, matrix_y)

    def __update(self, dw, db):
        self.w = self.w - self.params.eta * dw
        self.b = self.b - self.params.eta * db

    def inference(self, x):
        return self.__forward_batch(x)

    def train(self, data_reader, checkpoint=0.1):
        loss_history = TrainingHistory()
        loss_function = LossFunction(self.params.net_type)
        loss = 10
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
                    loss = self.__check_loss(loss_function, data_reader)
                    loss_history.add_loss_history(total_iteration, loss)
                    # print(epoch, iteration, loss)
                    if loss < self.params.eps:
                        break
            if loss < self.params.eps:
                break
        # loss_history.show_loss_history(self.params)
        print(self.w, self.b)
