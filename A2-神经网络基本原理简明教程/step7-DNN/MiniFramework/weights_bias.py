import numpy as np
from pathlib import Path
from .enum_def import InitialMethod


class WeightsBias(object):
    def __init__(self, n_input, n_output, init_method, eta):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.eta = eta
        self.initial_value_filename = str.format("w_{0}_{1}_{2}_init", self.num_input, self.num_output,
                                                 self.init_method.name)
        self.init_file = None
        self.db = None
        self.dw = None
        self.w = None
        self.b = None

    def initialize_weights(self, folder, create_new):
        self.init_file = str.format("{0}/{1}.npz", folder, self.initial_value_filename)
        if create_new:
            self.__create_new()
        else:
            self.__load_existing_parameters()
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)

    def __create_new(self):
        self.w, self.b = self.initial_parameters(self.num_input, self.num_output, self.init_method)
        self.__save_initial_value()

    def __load_existing_parameters(self):
        init_file = Path(self.init_file)
        if init_file.exists():
            self.__load_initial_value()
        else:
            self.__create_new()

    def update(self):
        self.w = self.w - self.eta * self.dw
        self.b = self.b - self.eta * self.db

    @staticmethod
    def initial_parameters(num_input, num_output, method):
        w = None
        if method == InitialMethod.Zero:
            w = np.zeros((num_input, num_output))
        elif method == InitialMethod.Normal:
            w = np.random.normal(size=(num_input, num_output))
        elif method == InitialMethod.Xavier:
            w = np.random.uniform(-np.sqrt(6 / (num_input + num_input)), np.sqrt(6 / (num_input + num_output)),
                                  size=(num_input, num_output))
        elif method == InitialMethod.MSRA:
            w = np.random.normal(0, np.sqrt(2 / num_output), size=(num_input, num_output))

        b = np.zeros((1, num_output))
        return w, b

    def __save_initial_value(self):
        np.savez(self.init_file, weights=self.w, bias=self.b)

    def save_result_value(self, folder, name):
        file_name = str.format("{0}/{1}.npz", folder, name)
        np.savez(file_name, weights=self.w, bias=self.b)

    def __load_initial_value(self):
        self.__load_value(self.init_file)

    def load_result_value(self, folder, name):
        file_name = str.format("{0}/{1}.npz", folder, name)
        self.__load_value(file_name)

    def __load_value(self, file_name):
        data = np.load(file_name)
        self.w = data["weights"]
        self.b = data["bias"]
