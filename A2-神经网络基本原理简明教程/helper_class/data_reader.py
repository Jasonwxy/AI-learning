import numpy as np
from pathlib import Path


class DataReader(object):
    def __init__(self, data_file):
        self.y_norm = None
        self.x_norm = None
        self.train_file_name = data_file
        self.num_train = 0
        self.x_train = None  # normalized x, if not normalized, same as XRaw 归一化X
        self.y_train = None  # normalized x, if not normalized, same as YRaw 归一化Y
        self.x_raw = None  # raw x 原始x
        self.y_raw = None  # raw y 原始y

        # read data from file

    def read_data(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.x_raw = data["data"]
            self.y_raw = data["label"]
            self.num_train = self.x_raw.shape[0]
            self.x_train = self.x_raw
            self.y_train = self.y_raw
        else:
            raise Exception("Cannot find train file!!!")
        # end if

    def normalize_x(self):
        x_new = np.zeros(self.x_raw.shape)
        num_feature = self.x_raw.shape[1]
        self.x_norm = np.zeros((num_feature, 2))
        # 按列归一化,即所有样本的同一特征值分别做归一化
        for i in range(num_feature):
            col_i = self.x_raw[:, i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            self.x_norm[i, 0] = min_value
            self.x_norm[i, 1] = max_value - min_value
            new_col = (col_i - self.x_norm[i, 0]) / self.x_norm[i, 1]
            x_new[:, i] = new_col
        self.x_train = x_new

    def normalize_predicate_data(self, x):
        x_new = np.zeros(x.shape)
        for i in range(x.shape[1]):
            col_i = x[:, i]
            x_new[:, i] = (col_i - self.x_norm[i, 0]) / self.x_norm[i, 1]
        return x_new

    def normalize_y(self):
        self.y_norm = np.zeros((1, 2))
        max_value = np.max(self.y_raw)
        min_value = np.min(self.y_raw)
        self.y_norm[0, 0] = min_value
        self.y_norm[0, 1] = max_value - min_value
        y_new = (self.y_raw - self.y_norm[0, 0]) / self.y_norm[0, 1]
        self.y_train = y_new

        # get batch training data

    def get_single_train_sample(self, iteration):
        x = self.x_train[iteration]
        y = self.y_train[iteration]
        return x, y

    # get batch training data
    def get_batch_train_samples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_x = self.x_train[start:end, :]
        batch_y = self.y_train[start:end, :]
        return batch_x, batch_y

    def get_whole_train_samples(self):
        return self.x_train, self.y_train

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        xp = np.random.permutation(self.x_train)
        np.random.seed(seed)
        yp = np.random.permutation(self.y_train)
        self.x_train = xp
        self.y_train = yp
