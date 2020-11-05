from helper_class2.mnist_image_data_reader import MnistImageDataReader
from helper_class2.hyper_parameters3 import HyperParameters
from helper_class2.enum_def import NetType, InitialMethod
from helper_class2.neural_net3 import NeuralNet

if __name__ == '__main__':
    reader = MnistImageDataReader(mode='vector')
    reader.read_data()
    reader.normalize_x()
    reader.normalize_y(NetType.MultipleClassifier, base=0)
    reader.shuffle()
    reader.generate_validation_set(k=12)

    n_input = reader.num_feature
    n_hidden1 = 64
    n_hidden2 = 16
    n_output = reader.num_category
    eta, batch_size, max_epoch, eps = 0.2, 128, 20, 0.01

    hp = HyperParameters(n_input, n_hidden1, n_hidden2, n_output, eta, max_epoch, batch_size, eps,
                         NetType.MultipleClassifier, InitialMethod.Xavier)

    net = NeuralNet(hp, 'mnist_64_16')
    # net.load_result()
    net.train(reader, 0.5, True)
    net.show_training_history(x='epoch')
