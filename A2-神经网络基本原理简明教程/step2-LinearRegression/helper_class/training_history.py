from matplotlib import pyplot as plt


class TrainingHistory(object):
    def __init__(self):
        self.iteration = []
        self.loss_history = []
        self.w_history = []
        self.b_history = []

    def add_loss_history(self, iteration, loss, w, b):
        self.iteration.append(iteration)
        self.loss_history.append(loss)
        self.w_history.append(w)
        self.b_history.append(b)

    def get_last(self):
        return self.iteration[-1], self.w_history[-1], self.b_history[-1]

    def show_loss_history(self, params, x_min=None, x_max=None, y_min=None, y_max=None):
        plt.plot(self.iteration, self.loss_history)
        title = params.to_string()
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        if not x_min and not y_min:
            plt.axis([x_min, x_max, y_min, y_max])
        plt.show()
