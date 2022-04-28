from .enum_def import NetType, InitialMethod


class HyperParameters(object):
    def __init__(self, eta=0.1, max_epoch=10000, batch_size=5, net_type=NetType.Fitting,
                 init_method=InitialMethod.Xavier, stopper=None):
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.stop = stopper
        self.net_type = net_type
        self.init_method = init_method

    def to_string(self):
        return str.format("bz:{0},eta:{1},init:{2}", self.batch_size, self.eta, self.init_method)
