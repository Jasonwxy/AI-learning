from helper_class2.enum_def import NetType, InitialMethod


class HyperParameters(object):
    def __init__(self, n_input, n_hidden, n_output, eta=0.1, max_epoch=10000, batch_size=5, eps=0.1,
                 net_type=NetType.Fitting, init_method=InitialMethod.Xavier):
        self.num_input = n_input
        self.num_hidden = n_hidden
        self.num_output = n_output

        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps
        self.net_type = net_type
        self.init_method = init_method

    def to_string(self):
        return str.format("bz:{0},eta:{1},ne:{2}", self.batch_size, self.eta, self.num_hidden)
