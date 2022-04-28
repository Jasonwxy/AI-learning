import numpy as np
from helper_class1.data_reader import DataReader

file_name = '../../../ai-edu/A-基础教程/A2-神经网络基本原理简明教程/data/ch05.npz'

if __name__ == "__main__":
    reader = DataReader(file_name)
    reader.read_data()
    matrix_x, matrix_y = reader.get_whole_train_samples()
    num_example = matrix_x.shape[0]
    one = np.ones((num_example, 1))
    x = np.column_stack((one, (matrix_x[0:num_example, :])))

    # a = np.dot(x.T, x)
    # b = np.asmatrix(a)
    # c = np.linalg.inv(b)
    # d = np.dot(c, x.T)
    # e = np.dot(d, matrix_y)
    result = np.dot(np.dot(np.linalg.inv(np.asmatrix(np.dot(x.T, x))), x.T), matrix_y)
    b = result[0, 0]
    w1 = result[1, 0]
    w2 = result[2, 0]
    print("w1=", w1)
    print("w2=", w2)
    print("b=", b)

    z = w1 * 15 + w2 * 93 + b
    print("z=", z)
