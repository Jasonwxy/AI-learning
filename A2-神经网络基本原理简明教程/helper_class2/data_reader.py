import numpy as np
from pathlib import Path


class DataReader(object):
    def __init__(self, train_file, test_file):
        self.y_norm = None
        self.x_norm = None
        self.train_file_name = train_file
        self.test_file_name = test_file
        self.num_train = 0  # num of training examples
        self.num_test = 0  # num of test examples
        self.num_validation = 0  # num of validation examples
        self.num_feature = 0  # num of features
        self.num_category = 0  # num of categories
        self.x_train = None  # training feature set
        self.y_train = None  # training label set
        self.x_test = None  # test feature set
        self.y_test = None  # test label set
        self.x_train_raw = None  # training feature set before normalization
        self.y_train_raw = None  # training label set before normalization
        self.x_test_raw = None  # test feature set before normalization
        self.y_test_raw = None  # test label set before normalization
        self.x_dev = None  # validation feature set
        self.y_dev = None  # validation label set

    # read data from file
    def read_data(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.x_train_raw = data["data"]
            self.y_train_raw = data["label"]
            assert (self.x_train_raw.shape[0] == self.y_train_raw.shape[0])
            self.num_train = self.x_train_raw.shape[0]
            self.num_feature = self.x_train_raw.shape[1]
            self.num_category = len(np.unique(self.y_train_raw))
            self.x_train = self.x_train_raw
            self.y_train = self.y_train_raw
        else:
            raise Exception("Cannot find train file!!!")

        test_file = Path(self.test_file_name)
        if test_file.exists():
            data = np.load(self.test_file_name)
            self.x_test_raw = data["data"]
            self.y_test_raw = data["label"]
            assert (self.x_test_raw.shape[0] == self.y_test_raw.shape[0])
            self.num_test = self.x_test_raw.shape[0]
            self.x_test = self.x_test_raw
            self.y_test = self.y_test_raw
            self.x_dev = self.x_test
            self.y_dev = self.y_test

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

    def to_one_hot(self, num_category, base=0):
        count = self.y_raw.shape[0]
        y = np.zeros((count, num_category))
        for i in range(count):
            n = int(self.y_raw[i, 0])
            y[i, n - base] = 1
        self.y_train = y

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
