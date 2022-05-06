import numpy as np
from pathlib import Path
from .enum_def import NetType


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
        if not train_file.exists():
            raise Exception("Cannot find train file!!!")

        data = np.load(self.train_file_name)
        self.x_train_raw = data["data"]
        self.y_train_raw = data["label"]
        assert (self.x_train_raw.shape[0] == self.y_train_raw.shape[0])
        self.num_train = self.x_train_raw.shape[0]
        self.num_feature = self.x_train_raw.shape[1]
        self.num_category = len(np.unique(self.y_train_raw))
        self.x_train = self.x_train_raw
        self.y_train = self.y_train_raw
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
        x_merge = np.vstack((self.x_train_raw, self.x_test_raw))
        x_merge_norm = self.__normalize_x(x_merge)
        self.x_train = x_merge_norm[0:self.num_train, :]
        self.x_test = x_merge_norm[self.num_train:, :]

    def __normalize_x(self, raw_data):
        temp_x = np.zeros_like(raw_data)
        self.x_norm = np.zeros((2, self.num_feature))
        # 按列归一化,即所有样本的同一特征值分别做归一化
        for i in range(self.num_feature):
            col_i = raw_data[:, i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            self.x_norm[0, i] = min_value
            self.x_norm[1, i] = max_value - min_value
            new_col = (col_i - self.x_norm[0, i]) / self.x_norm[1, i]
            temp_x[:, i] = new_col
        return temp_x

    def normalize_y(self, net_type, base=0):
        if net_type == NetType.Fitting:
            y_merge = np.vstack((self.y_train_raw, self.y_test_raw))
            y_merge_norm = self.__normalize_y(y_merge)
            self.y_train = y_merge_norm[0:self.num_train, :]
            self.y_test = y_merge_norm[self.num_train:, :]
        elif net_type == NetType.BinaryClassifier:
            self.y_train = self.__to_zero_one(self.y_train_raw)
            self.y_test = self.__to_zero_one(self.y_test_raw)
        elif net_type == NetType.MultipleClassifier:
            self.y_train = self.__to_one_hot(self.y_train_raw, base)
            self.y_test = self.__to_one_hot(self.y_test_raw, base)

    def __normalize_y(self, raw_data):
        assert (raw_data.shape[1] == 1)
        self.y_norm = np.zeros((2, 1))
        max_value = np.max(raw_data)
        min_value = np.min(raw_data)
        self.y_norm[0, 0] = min_value
        self.y_norm[1, 0] = max_value - min_value
        return (raw_data - self.y_norm[0, 0]) / self.y_norm[1, 0]

    def __to_one_hot(self, y, base=0):
        count = y.shape[0]
        tmp_y = np.zeros((count, self.num_category))
        for i in range(count):
            n = int(y[i, 0])
            tmp_y[i, n - base] = 1
        return tmp_y

    @staticmethod
    def __to_zero_one(y, positive_label=1, negative_label=0, positive_value=1, negative_value=0):
        tmp_y = np.zeros_like(y)
        for i in range(y.shape[0]):
            if y[i, 0] == positive_label:
                tmp_y[i, 0] = positive_value
            elif y[i, 0] == negative_label:
                tmp_y[i, 0] = negative_value
        return tmp_y

    def de_normalize_y(self, predict_value):
        return predict_value * self.y_norm[1, 0] + self.y_norm[0, 0]

    def normalize_predicate_data(self, x):
        x_new = np.zeros(x.shape)
        for i in range(x.shape[0]):
            col_i = x[i, :]
            x_new[i, :] = (col_i - self.x_norm[0, i]) / self.x_norm[1, i]
        return x_new

    def generate_validation_set(self, k=10):
        self.num_validation = int(self.num_train / k)
        self.num_train = self.num_train - self.num_validation
        self.x_dev = self.x_train[0:self.num_validation]
        self.y_dev = self.y_train[0:self.num_validation]
        self.x_train = self.x_train[self.num_validation:]
        self.y_train = self.y_train[self.num_validation:]

    def get_validation_set(self):
        return self.x_dev, self.y_dev

    def get_test_set(self):
        return self.x_test, self.y_test

    # get batch training data
    def get_batch_train_samples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_x = self.x_train[start:end, :]
        batch_y = self.y_train[start:end, :]
        return batch_x, batch_y

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
