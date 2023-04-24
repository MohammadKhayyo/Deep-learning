import numpy as np
import sys


def preprocess_data(filename, x_0=1):
    """This function reads the file and converts its data into a matrix,then it shuffles these lines ,and then separates
    the training and labels. then return The training and labels. """
    try:
        with open(filename) as data_file:
            inputs = np.loadtxt(data_file, dtype=str, delimiter=',')
            np.random.shuffle(inputs)
            x_0_np = np.zeros((inputs.shape[0], 1), dtype=int) + x_0
            all_data = np.append(x_0_np, inputs, axis=1)
            expected_y_str = all_data[:, -1]
            data_for_train_str = all_data[:, :-1]
            data_for_train = data_for_train_str.astype(int)
            expected_y = np.array([1 if i == 'positive' else 0 for i in expected_y_str])
    except Exception as e:
        print(f'error : {e}')
        exit(1)
    return data_for_train, expected_y


def train(data_for_train, expected_y, learning_rate=0.0001, threshold=0.5, iterations_num=1500):
    """the function trains the model according to randomly chosen weights value between (-1, 1) and the data."""
    weights = np.random.uniform(low=0.0, high=1.0, size=data_for_train.shape[1])
    for _ in range(iterations_num):
        for train_num, x_i in enumerate(data_for_train):
            Sum = np.dot(x_i, weights)
            est_y = np.where(Sum > threshold, 1, 0)
            Error = expected_y[train_num] - est_y
            Correction = learning_rate * Error
            Delta_weights = Correction * x_i
            weights += Delta_weights
    return weights


def predict(data_for_train, expected_y, weights, threshold=0.5):
    """the function predict the result of the given inputs and calculate the accuracy."""
    Sum = np.dot(data_for_train, weights)
    est_y = np.where(Sum > threshold, 1, 0)
    accuracy = np.sum(est_y == expected_y) / expected_y.shape[0]
    return accuracy * 100, est_y


if __name__ == "__main__":
    file_name = 'data_ex1.txt' if len(sys.argv) == 1 else sys.argv[1]
    _data_for_train, _expected_y = preprocess_data(file_name)
    _weights = train(_data_for_train, _expected_y, learning_rate=0.0001,
                     threshold=0.5, iterations_num=1000)
    _accuracy, _est_y = predict(_data_for_train, _expected_y, _weights, threshold=0.5)
    print('weights:', _weights)
    print(f'accuracy: {_accuracy}%')
