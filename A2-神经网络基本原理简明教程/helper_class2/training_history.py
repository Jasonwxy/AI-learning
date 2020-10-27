import pickle
from matplotlib import pyplot as plt


class TrainingHistory(object):
    def __init__(self):
        self.loss_train = []
        self.accuracy_train = []
        self.iteration_seq = []
        self.epoch_seq = []
        self.loss_val = []
        self.accuracy_val = []

    def add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld):
        self.iteration_seq.append(total_iteration)
        self.epoch_seq.append(epoch)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_val.append(accuracy_vld)
        return False

    def show_loss_history(self, params, x_min=None, x_max=None, y_min=None, y_max=None):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        p1, = plt.plot(self.epoch_seq, self.loss_val)
        p2, = plt.plot(self.epoch_seq, self.loss_train)
        plt.legend([p1, p2], ["validation", "train"])
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        if x_min is not None or x_max is not None or y_max is not None or y_min is not None:
            plt.axis([x_min, x_max, y_min, y_max])

        plt.subplot(1, 2, 2)
        p1, = plt.plot(self.epoch_seq, self.accuracy_val)
        p2, = plt.plot(self.epoch_seq, self.accuracy_train)
        plt.legend([p1, p2], ["validation", "train"])
        plt.title("Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        title = params.to_string()
        plt.suptitle(title)
        plt.show()

    def show_loss_history_4(self, axes, params, x_min=None, x_max=None, y_min=None, y_max=None):
        axes.plot(self.epoch_seq, self.loss_val)
        axes.plot(self.epoch_seq, self.loss_train)
        title = params.to_string()
        axes.set_title(title)
        axes.set_xlabel("epoch")
        axes.set_ylabel("loss")
        if not x_min and not y_min:
            plt.axis([x_min, x_max, y_min, y_max])

    def dump(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            lh = pickle.load(f)
            return lh
