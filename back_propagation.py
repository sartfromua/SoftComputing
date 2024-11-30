# Variant 2
import random

import numpy as np
import random
import time
import functools
from numpy import array as arr
from itertools import product
from matplotlib import pyplot as plt
from prettytable import PrettyTable


number_of_neuron = 1
number_of_layers = 1


def timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        print(f"{func.__name__} took {runtime:.4f} secs")
        return result
    return _wrapper


def function(x):
    return 1 / (1 + np.exp(-x))
    # ReLU
    # return np.maximum(0, x)
    # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def derivative(x):
    return x * (1 - x)
    # ReLU
    # return (x > 0).astype(float)

    # return 4 * np.exp(2*x) / (np.exp(2*x)+1)**2


def true_function(x: arr):
    # return arr([np.sum(np.square(x_i)) for x_i in x])
    return arr([np.sum(np.sin(x_i)) for x_i in x])


def normalize(array: arr):
    # MinMax normalization
    max_val = np.max(array)
    min_val = np.min(array)
    array = (array - min_val) / (max_val - min_val)
    # array = 2 * (array - min_val) / (max_val - min_val) - 1
    return array


def denormalize(array, min_val, max_val):
    array = arr(array)
    array = array * (max_val - min_val) + min_val
    # array = (array - 1) / 2 * (max_val - min_val) + min_val
    return array


def sample_split(x_values: arr, y_values: arr, test_ratio=0.25):
    length = int(x_values.shape[0] * test_ratio)
    x_copied = np.copy(x_values)
    y_copied = np.copy(y_values)
    assert x_copied.shape[0] == y_copied.shape[0]
    # random.shuffle(x_copied)
    # random.shuffle(y_copied)
    x_test_split = x_copied[:length]
    y_test_split = y_copied[:length]
    x_train_split = x_copied[length:]
    y_train_split = y_copied[length:]
    assert x_test_split.shape[0] == y_test_split.shape[0]
    assert x_test_split.shape[0] + x_train_split.shape[0] == x_copied.shape[0]
    return x_train_split, y_train_split, x_test_split, y_test_split


def loss_function(output, ground_truth):
    # MSE
    # loss = np.mean(np.square(ground_truth - output)) / np.mean(ground_truth**2)
    loss = np.sum(np.square(ground_truth - output))
    return loss


def get_x_values(n=27):
    x_start = arr([5, 4, 3])
    if n <= 27:
        difference = sorted(list(product(range(-1, 2), repeat=3)), reverse=True)
    else:
        difference = sorted(list(product(range(-2, 3), repeat=3)), reverse=True)
    additional_x = [np.array(x_start) + np.array(list(difference[i])) for i in range(n)]
    x_values = arr(list(map(lambda value: list(value), additional_x)), dtype=np.float32)
    x_values = np.reshape(x_values, (x_values.size // 3, 3))
    return x_values


def get_ground_truth(x_values: arr):
    true_res = true_function(x_values)
    aver = np.mean(true_res)
    ground_truth = np.zeros(x_values.shape[0] * 2)
    ground_truth = np.reshape(ground_truth, (x_values.shape[0], 2))
    for i in range(x_values.shape[0]):
        ground_truth[i, 0] = true_res[i]
        ground_truth[i, 1] = 1 if true_res[i] > aver else 0
    return ground_truth


class Neuron:
    def __init__(self, input_size):
        global number_of_neuron
        self.name = number_of_neuron
        number_of_neuron += 1
        self.weights = arr([random.uniform(0, 1) for i in range(input_size)])
        # self.weights = np.random.randn(input_size) / np.sqrt(input_size)

        self.output = None

    def activate(self, inputs: arr):
        # sigmoid
        a = np.dot(inputs, self.weights)
        self.output = function(a)
        return self.output

    def __str__(self):
        return f"Neuron #{self.name}, weights: {self.weights}, output: {self.output}"


class Layer:
    def __init__(self, input_size, num_neurons):
        self.neurons = arr([Neuron(input_size) for _ in range(num_neurons)])
        global number_of_layers
        self.name = number_of_layers
        number_of_layers += 1
        self.output = None
        self.error = None
        self.delta = None

    def forward(self, inputs: arr):
        self.output = arr([neuron.activate(inputs) for neuron in self.neurons])
        return self.output

    def __str__(self):
        return "Layer #" + str(self.name) + "\n" + str([str(neuron) for neuron in self.neurons])


class Network:
    def __init__(self, layers: list):
        self.layers = layers
        self.loss_changes = []

    def print_table(self, x: arr, ground_truth: arr, x_min, x_max, y_min, y_max, denorm=True):
        x_values = np.copy(x)
        gt = np.copy(ground_truth)
        table = PrettyTable(["x1", "x2", "x3", "d1", "d2", "T1", "T2"])
        table.float_format = '0.4'
        output = arr([self.predict(x) for x in x_values])

        if denorm:
            for i in range(x_values.shape[1]):
                x_values[:, i] = denormalize(x_values[:, i], x_min[i], x_max[i])
            output[:, 0] = denormalize(output[:, 0], y_min, y_max)
            gt[:, 0] = denormalize(gt[:, 0], y_min, y_max)
        # output[:, 1] = output[:, 1].round()

        for i in range(x_values.shape[0]):
            table.add_row(
                [*x_values[i], output[i, 0], output[i, 1], gt[i, 0], gt[i, 1]]
            )
        print(table)
        print("Mean Squared Error:", loss_function(output, gt))

    def show_neurons(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                print(f"Neuron {neuron.name} Weights: {neuron.weights}")

    def predict(self, x: arr):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_table(self, X):
        output = arr([self.predict(x) for x in X])
        return output

    def backward(self, x: arr, ground_truth: arr, learning_rate: float = 0.05):
        output = self.predict(x)

        self.layers[-1].error = ground_truth - output
        self.layers[-1].delta = self.layers[-1].error * derivative(output)

        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # weights = np.array([neuron.weights for neuron in next_layer.neurons])
            # layer.error = np.dot(next_layer.delta, weights)
            layer.error = np.dot(next_layer.delta, np.array([neuron.weights for neuron in next_layer.neurons]))
            # weights = np.array([neuron.weights for neuron in next_layer.neurons]).T
            # layer.error = np.dot(next_layer.delta, weights)

            # error = [0 for i in range(layer.output.shape[0])]
            # for i in range(layer.neurons.shape[0]):
            #     for k in range(next_layer.neurons.shape[0]):
            #         error[k] += next_layer.delta[k] * next_layer.neurons[k].weights[i]
            # layer.error = arr(error[0])

            layer.delta = layer.error * derivative(layer.output)

        for i in range(len(self.layers)):
            # print(self.layers[i])
            # print("layer.error = np.dot(next_layer.delta, weights) * derivative(layer.output)")
            # print(self.layers[i].delta)
            # print(self.layers[i].error)
            layer = self.layers[i]
            inputs = x if i == 0 else self.layers[i - 1].output
            for j, neuron in enumerate(layer.neurons):
                neuron.weights += learning_rate * layer.delta[j] * inputs

    @timer
    def train(self, X, ground_truth, epochs, learning_rate=0.05, show_result=True):
        # print("Neurons before training")
        # self.show_neurons()
        for epoch in range(epochs):
            for x_i, y_i in zip(X, ground_truth):
                self.backward(x_i, y_i, learning_rate)
            output = arr([self.predict(x) for x in X])
            self.loss_changes.append(loss_function(output, ground_truth))
        # print("Neurons after training")
        # self.show_neurons()

        if show_result:
            plt.plot(self.loss_changes)
            plt.title("Loss changes")
            plt.show()
        print("Best result")
        print("MSE: ", end="")
        print(sorted(self.loss_changes)[0])


if __name__ == "__main__":
    input_size = 3
    num_neurons = 3
    output_size = 2

    model = Network([
        Layer(input_size, num_neurons),
        Layer(num_neurons, output_size)]
    )

    x = get_x_values(n=27)
    y = get_ground_truth(x)
    print(x.shape)

    print("Values before normalization")
    table = PrettyTable(["x1", "x2", "x3", "T1", "T2"])
    table.float_format = '0.4'
    table.add_rows([[*x[i], *y[i]] for i in range(x.shape[0])])
    print(table)

    # Normalization
    x_norm = np.copy(x)
    y_norm = np.copy(y)
    x_norm[:, 0] = normalize(x_norm[:, 0])
    x_norm[:, 1] = normalize(x_norm[:, 1])
    x_norm[:, 2] = normalize(x_norm[:, 2])
    y_norm[:, 0] = normalize(y_norm[:, 0])

    x_min = [np.min(x[:, i]) for i in range(x.shape[1])]
    x_max = [np.max(x[:, i]) for i in range(x.shape[1])]
    y_min = np.min(y[:, 0])
    y_max = np.max(y[:, 0])

    print("Values after normalization")
    table.clear_rows()
    table.float_format = '0.4'
    for i in range(x.shape[0]):
        table.add_row([*x_norm[i], *y_norm[i]])
    print(table)

    x_train, y_train, x_test, y_test = sample_split(x_norm, y_norm, test_ratio=0.25)

    # Tables for default model
    # print("\nBefore training")
    # model.print_table(x_train, y_train, x_min, x_max, y_min, y_max, denorm=False)
    #
    # print(*model.layers[0].neurons[0].weights)
    # model.train(x_train, y_train, 1000, learning_rate=0.15)
    #
    # print("\nAfter training")
    # print(*model.layers[0].neurons[0].weights)
    # model.print_table(x_train, y_train, x_min, x_max, y_min, y_max, denorm=False)

    # print("\nTrain table")
    # model.print_table(x_train, y_train, x_min, x_max, y_min, y_max)

    # print("\nTest table")
    # model.print_table(x_test, y_test, x_min, x_max, y_min, y_max)

    # Correlation check if near 1 then correlation is seen
    # outputs = np.array([model.predict(x) for x in x_test])
    # correlation = np.corrcoef(outputs, rowvar=False)
    # print("Correlation between output neurons:\n", correlation)

    models = list()
    models.append((model, "1 layer: 3 neurons"))
    models.append((Network([
        Layer(input_size, 6),
        Layer(6, output_size)
    ]), "1 layer: 6 neurons"))
    models.append((Network([
        Layer(input_size, 12),
        Layer(12, output_size)
    ]), "1 layer: 12 neurons"))

    # models = models[2:3]

    for mod, name in models:
        mod.train(x_train, y_train, 300, learning_rate=0.1, show_result=False)
        # results.append((mod.loss_changes, name))
        plt.plot(mod.loss_changes, label=name)
        plt.legend(loc='best')
    plt.show()

    for mod, name in models:
        print("\n", name)
        print("Train table")
        mod.print_table(x_train, y_train, x_min, x_max, y_min, y_max, denorm=False)
        print("\nTest table")
        mod.print_table(x_test, y_test, x_min, x_max, y_min, y_max, denorm=True)
