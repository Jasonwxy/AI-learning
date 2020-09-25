import numpy as np
from helper_class.training_history import TrainingHistory
from helper_class.data_reader import DataReader

date_reader = DataReader("")


class NeuralNet(object):

    def __init__(self, params):
        self.params = params
        self.w = 0
        self.b = 0

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

    def train(self, data_reader):
        loss_history = TrainingHistory()

        if self.params.batch_size == -1:
            self.params.batch_size = data_reader.num_train
        max_iteration = int(data_reader.num_train / self.params.batch_size)
        for epoch in range(self.params.max_epoch):
            print("epoch=%d" % epoch)
            data_reader.suffle()
            for iteration in range(max_iteration):
                batch_x, batch_y = date_reader.get_batch_train_samples(self.params.batch_size, iteration)
