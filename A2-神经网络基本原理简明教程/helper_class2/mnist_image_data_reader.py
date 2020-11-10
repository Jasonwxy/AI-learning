import numpy as np
import struct
from helper_class2.data_reader import DataReader

train_image_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/train-images-10'
train_label_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/train-labels-10'
test_image_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/test-images-10'
test_label_file = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/test-labels-10'


class MnistImageDataReader(DataReader):
    def __init__(self, mode='image', train_file=None, test_file=None):
        super().__init__(train_file, test_file)
        self.train_image_file = train_image_file
        self.train_label_file = train_label_file
        self.test_image_file = test_image_file
        self.test_label_file = test_label_file
        self.mode = mode

    def read_data(self):
        self.__read_data()

    def read_less_data(self, count):
        self.__read_data(count)

    def __read_data(self, count=-1):
        self.x_train_raw = self.__read_image_file(train_image_file)
        self.y_train_raw = self.__read_label_file(train_label_file)
        self.x_test_raw = self.__read_image_file(test_image_file)
        self.y_test_raw = self.__read_label_file(test_label_file)
        if count != -1:
            self.x_train_raw = self.x_train_raw[0:count]
            self.y_train_raw = self.y_train_raw[0:count]
        self.num_category = np.unique(self.y_train_raw).shape[0]
        self.num_train = self.x_train_raw.shape[0]
        self.num_test = self.x_test_raw.shape[0]
        if self.mode == 'vector':
            self.num_feature = 784

    @staticmethod
    def __read_image_file(image_file_name):
        with open(image_file_name, 'rb') as f:
            bin_data = f.read()
            offset = 0
            fmt_header = '>iiii'
            magic_num, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
            image_size = num_rows * num_cols
            fmt_image = '>' + str(image_size) + 'B'
            offset += struct.calcsize(fmt_header)
            image_data = np.zeros((num_images, 1, num_rows, num_cols))
            for i in range(num_images):
                image_data[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape(
                    (1, num_rows, num_cols))
                offset += struct.calcsize(fmt_image)
        return image_data

    @staticmethod
    def __read_label_file(label_file_name):
        with open(label_file_name, 'rb') as f:
            bin_data = f.read()
            offset = 0
            fmt_header = '>ii'
            magic_num, num_labels = struct.unpack_from(fmt_header, bin_data, offset)
            offset += struct.calcsize(fmt_header)
            fmt_label = '>B'
            labels_data = np.zeros((num_labels, 1))
            for i in range(num_labels):
                labels_data[i] = np.array(struct.unpack_from(fmt_label, bin_data, offset))[0]
                offset += struct.calcsize(fmt_label)
        return labels_data

    def normalize_x(self):
        self.x_train = self.__normalize_data(self.x_train_raw)
        self.x_test = self.__normalize_data(self.x_test_raw)

    @staticmethod
    def __normalize_data(raw_data):
        x_max = np.max(raw_data)
        x_min = np.min(raw_data)
        return (raw_data - x_min) / (x_max - x_min)

    def get_batch_train_samples(self, batch_size, iteration):
        batch_x, batch_y = super().get_batch_train_samples(batch_size, iteration)
        if self.mode == 'vector':
            return batch_x.reshape(-1, 784), batch_y
        return batch_x, batch_y

    def get_validation_set(self):
        if self.mode == 'vector':
            return self.x_dev.reshape(self.num_validation, -1), self.y_dev
        return self.x_dev, self.y_dev

    def get_test_set(self):
        if self.mode == 'vector':
            return self.x_test.reshape(self.num_test, -1), self.y_test
        return self.x_test, self.y_test

    def get_batch_test_samples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_x = self.x_train[start:end]
        batch_y = self.y_train[start:end]

        if self.mode == 'vector':
            return batch_x.reshape(batch_size, -1), batch_y
        return batch_x, batch_y
