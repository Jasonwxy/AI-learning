import numpy as np
import math
import os
import time
from helper_class2.training_history import TrainingHistory
from helper_class2.enum_def import NetType
from helper_class2.classifier_function import Logistic, SoftMax
from helper_class2.loss_function import LossFunction
from helper_class2.weights_bias import WeightsBias
from helper_class2.activator_function import Sigmoid, Tanh


class NeuralNet(object):

    def __init__(self, hp, model_name):
        self.hp = hp
        self.model_name = model_name
        self.sub_folder = os.getcwd() + "\\" + self.__create_sub_folder()
        self.loss_trace = TrainingHistory()
        self.loss_func = LossFunction(self.hp.net_type)

        self.wb1 = WeightsBias(self.hp.num_input, self.hp.num_hidden1, self.hp.init_method, self.hp.eta)
        self.wb1.initialize_weights(self.sub_folder, False)
        self.wb2 = WeightsBias(self.hp.num_hidden1, self.hp.num_hidden2, self.hp.init_method, self.hp.eta)
        self.wb2.initialize_weights(self.sub_folder, False)
        self.wb3 = WeightsBias(self.hp.num_hidden2, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb3.initialize_weights(self.sub_folder, False)

        self.a1 = None
        self.a2 = None
        self.a3 = None

    def __create_sub_folder(self):
        if self.model_name is not None:
            path = self.model_name.strip()
            path = path.rstrip("\\")
            if not os.path.exists(path):
                os.mkdir(path)
            return path

    def forward(self, batch_x):
        z1 = np.dot(batch_x, self.wb1.w) + self.wb1.b
        self.a1 = Sigmoid().forward(z1)
        z2 = np.dot(self.a1, self.wb2.w) + self.wb2.b
        self.a2 = Tanh().forward(z2)
        z3 = np.dot(self.a2, self.wb3.w) + self.wb3.b

        if self.hp.net_type == NetType.BinaryClassifier:
            self.a3 = Logistic().forward(z3)
        elif self.hp.net_type == NetType.MultipleClassifier:
            self.a3 = SoftMax().forward(z3)
        elif self.hp.net_type == NetType.Fitting:
            self.a3 = z3

    def backward(self, batch_x, batch_y):
        m = batch_x.shape[0]
        dz3 = self.a3 - batch_y
        self.wb3.dw = np.dot(self.a2.T, dz3) / m
        self.wb3.db = np.sum(dz3, axis=0, keepdims=True) / m

        da2 = np.dot(dz3, self.wb3.w.T)
        dz2, _ = Tanh().backward(None, self.a2, da2)
        self.wb2.dw = np.dot(self.a1.T, dz2) / m
        self.wb2.db = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.wb2.w.T)
        dz1, _ = Sigmoid().backward(None, self.a1, da1)
        self.wb1.dw = np.dot(batch_x.T, dz1)
        self.wb1.db = np.sum(dz1, axis=0, keepdims=True) / m

    def update(self):
        self.wb1.update()
        self.wb2.update()
        self.wb3.update()

    def inference(self, x):
        self.forward(x)
        return self.a3

    def train(self, data_reader, checkpoint, need_test):
        t0 = time.time()
        if self.hp.batch_size == -1:
            self.hp.batch_size = data_reader.num_train
        max_iteration = math.ceil(data_reader.num_train / self.hp.batch_size)
        checkpoint_iteration = int(max_iteration * checkpoint)
        need_stop = False
        for epoch in range(self.hp.max_epoch):
            # print("epoch=%d" % epoch)
            data_reader.shuffle()
            for iteration in range(int(max_iteration)):
                batch_x, batch_y = data_reader.get_batch_train_samples(self.hp.batch_size, iteration)
                self.forward(batch_x)
                self.backward(batch_x, batch_y)
                self.update()

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    need_stop = self.check_error_and_loss(data_reader, batch_x, batch_y, epoch, total_iteration)
                    # print(epoch, iteration, loss)
                    if need_stop:
                        break
            if need_stop:
                break
        self.save_result()
        t1 = time.time()
        print("time used:", t1 - t0)
        if need_test:
            print("testing...")
            accuracy = self.test(data_reader)
            print(accuracy)

    def check_error_and_loss(self, data_reader, batch_x, batch_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" % (epoch + 1, total_iteration + 1))
        self.forward(batch_x)
        loss_train = self.loss_func.check_loss(self.a3, batch_y)
        accuracy_train = self.__cal_accuracy(self.a3, batch_y)
        print("loss_train=%.6f, accuracy_train=%f" % (loss_train, accuracy_train))

        vld_x, vld_y = data_reader.get_validation_set()
        self.forward(vld_x)
        loss_vld = self.loss_func.check_loss(self.a3, vld_y)
        accuracy_vld = self.__cal_accuracy(self.a3, vld_y)
        print("loss_valid=%.6f, accuracy_valid=%f" % (loss_vld, accuracy_vld))

        need_stop = False
        self.loss_trace.add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld)
        if loss_vld <= self.hp.eps:
            need_stop = True
        return need_stop

    def __cal_accuracy(self, a, y):
        assert (a.shape == y.shape)
        m = a.shape[0]
        if self.hp.net_type == NetType.Fitting:
            var = np.var(y)
            mse = np.sum((a - y) ** 2) / m
            return 1 - mse / var
        elif self.hp.net_type == NetType.BinaryClassifier:
            b = np.round(a)
            r = (b == y)
            return np.sum(r) / m
        elif self.hp.net_type == NetType.MultipleClassifier:
            ra = np.argmax(a, axis=1)
            ry = np.argmax(y, axis=1)
            r = (ra == ry)
            return r.sum() / m

    def save_result(self):
        self.wb1.save_result_value(self.sub_folder, 'wb1')
        self.wb2.save_result_value(self.sub_folder, 'wb2')
        self.wb3.save_result_value(self.sub_folder, 'wb3')

    def load_result(self):
        self.wb1.load_result_value(self.sub_folder, 'wb1')
        self.wb2.load_result_value(self.sub_folder, 'wb2')
        self.wb3.load_result_value(self.sub_folder, 'wb2')

    def test(self, data_reader):
        x, y = data_reader.get_test_set()
        print(x.shape, y.shape)
        self.forward(x)
        return self.__cal_accuracy(self.a3, y)

    def show_training_history(self, x):
        self.loss_trace.show_loss_history(self.hp, x)

    def get_training_trace(self):
        return self.loss_trace

    def get_epoch_num(self):
        return self.loss_trace.get_epoch_num()

    def dump_loss_history(self, file_name):
        self.loss_trace.dump(file_name)
