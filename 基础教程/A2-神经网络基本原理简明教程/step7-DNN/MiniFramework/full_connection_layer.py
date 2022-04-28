import numpy as np
from .layer import Layer
from .weights_bias import WeightsBias


class FCLayer(Layer):

    def __init__(self, input_size, output_size, param):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = WeightsBias(self.input_size, self.output_size, param.init_method, param.eta)

    def initialize(self, folder):
        self.weights.initialize_weights(folder, False)

    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.x = inputs
        self.z = np.dot(self.x, self.weights.w) + self.weights.b
        return self.z

    def backward(self, delta_in, layer_idx):
        dz = delta_in
        m = self.x.shape[0]
        self.weights.dw = np.dot(self.x.T, dz) / m
        self.weights.db = np.sum(dz, axis=0, keepdims=True) / m
        if layer_idx == 0:
            return None

        delta_out = np.dot(dz, self.weights.w.T)

        if len(self.input_shape) > 2:
            return delta_out.reshape(self.input_shape)
        else:
            return delta_out

    def update(self):
        self.weights.update()

    def save_parameters(self, folder, name):
        self.weights.save_result_value(folder, name)

    def load_parameters(self, folder, name):
        self.weights.load_result_value(folder, name)
