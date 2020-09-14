import numpy as np
from pathlib import Path


class DataReader(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = None
        self.YTrain = None

    # read data from file
    def read_data(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XTrain = data["data"]
            self.YTrain = data["label"]
            self.num_train = self.XTrain.shape[0]
        else:
            raise Exception("Cannot find train file!!!")
        # end if

    # get batch training data
    def get_single_train_sample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x, y

    # get batch training data
    def get_batch_train_samples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_x = self.XTrain[start:end, :]
        batch_y = self.YTrain[start:end, :]
        return batch_x, batch_y

    def get_whole_train_samples(self):
        return self.XTrain, self.YTrain

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        xp = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        yp = np.random.permutation(self.YTrain)
        self.XTrain = xp
        self.YTrain = yp
