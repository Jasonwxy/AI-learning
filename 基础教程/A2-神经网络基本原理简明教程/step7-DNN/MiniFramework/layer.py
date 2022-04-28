class Layer(object):

    def __init__(self):
        self.input_shape = None
        self.x =None
        self.z = None

    def initialize(self,folder):
        pass

    def train(self, inputs, train=True):
        pass

    def forward(self, inputs):
        pass

    def backward(self,delta_in,arg):
        pass

    def update(self):
        pass

    def save_parameters(self, folder, name):
        pass

    def load_parameters(self, folder, name):
        pass
