import numpy as np
import pandas as pd
import random as r
from itertools import product
import matplotlib.pyplot as plt
from time import time


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dx_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def func(x):
    # return np.sum(np.sin(x))
    return np.tan(x[0]) + np.sin(x[1]) - np.sin(x[2])
    # return np.log(abs(np.cos(x[0]))) + np.tan(x[1]) + (1 / np.tan(x[2]))


def norm_array(arr):
    arr = np.array(arr)
    max_val = np.max(arr)
    min_val = np.min(arr)
    arr = (arr - min_val) / (max_val - min_val)
    return arr


def denorm_array(arr, min_val, max_val):
    arr = np.array(arr)
    arr = arr * (max_val - min_val) + min_val
    return arr


def predict(X, W1, W2):
    predictions = []
    for i in range(len(X)):
        xi = X[i]
        z_1 = np.dot(W1, xi)
        y_1 = sigmoid(z_1)
        z_2 = np.dot(W2, y_1)
        y_2 = sigmoid(z_2)
        predictions.append(y_2)
    return np.array(predictions)


columns = ['x1', 'x2', 'x3', 'd1', 'd2']
default_X = [1, 2, 3]
# default_X = [5, 7, 8]
difference = sorted(list(product([-1, 0, 1], repeat=3)), reverse=True)
additional_X = [np.array(default_X) + np.array(list(difference[i])) for i in range(20)]
X = list(map(lambda value: list(value), additional_X))
data = pd.DataFrame(X, columns=columns[:3])

d1 = [func(x) for x in X]
avg = np.sum(d1) / len(d1)
d2 = [1 if d1k > avg else 0 for d1k in d1]

data[columns[3]] = d1
data[columns[4]] = d2

data2 = data.copy()
for i in range(4):
    data2[columns[i]] = norm_array(data2[columns[i]])

# for i in range(4):
#     data2[columns[i]] = denorm_array(data2[columns[i]], np.min(data[columns[i]]), np.max(data[columns[i]]))
# for i in range(data2.shape[0]):
#     data2['d1'][i] = func(np.array(data2.loc[i, :'x3']))
# norm_avg = np.mean(data2['d1'])
# data2['d2'] = [1 if data2['d1'][i] > norm_avg else 0 for i in range(data2.shape[0])]

print('Датафрейм з оригінальними даними')
results_df = pd.DataFrame({
    'x1': data['x1'],
    'x2': data['x2'],
    'x3': data['x3'],
    'True d1': data['d1'],
    'True d2': data['d2']
})
print(results_df)

print('Датафрейм з нормованими даними')
results_df = pd.DataFrame({
    'x1': data2['x1'],
    'x2': data2['x2'],
    'x3': data2['x3'],
    'True d1': data2['d1'],
    'True d2': data2['d2']
})
print(results_df)

train_data = data2[:int(len(data2) * 0.8)]
test_data = data2[-int(len(data2) * 0.2):]

@timer_func
def back_propagation(X, d1, d2, l, p=3, k=2, epochs=500):
    W1 = np.random.uniform(0, 1, (l, p))
    W2 = np.random.uniform(0, 1, (k, l))

    step = 0.5
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            xi = X[i, :p]
            target = np.array([d1[i], d2[i]])

            z_1 = np.dot(W1, xi)
            y_1 = sigmoid(z_1)

            z_2 = np.dot(W2, y_1)
            y_2 = sigmoid(z_2)

            errors = target - y_2
            total_loss += np.mean(np.square(errors))

            delta_output = errors * dx_sigmoid(z_2)
            delta_hidden = np.dot(W2.T, delta_output) * dx_sigmoid(z_1)

            W2 += step * np.outer(delta_output, y_1)
            W1 += step * np.outer(delta_hidden, xi)

        loss_history.append(total_loss / len(X))

    return W1, W2, loss_history


X_train = train_data.iloc[:, :3].values
d1_train = train_data['d1'].values
d2_train = train_data['d2'].values

hidden_layer_neurons = [2, 3, 4, 8, 10, 12, 20, 40]
epochs = 1000

losses_by_neurons = {}
result_data = {'HLN': [2, 4, 8, 10, 12, 20, 40],
               'W1': [],
               'W2': [],
               'LOSS': []}

X_test = test_data.iloc[:, :3].values
d1_test = test_data['d1'].values
d2_test = test_data['d2'].values

for l in hidden_layer_neurons:
    print(f"Training with {l} neurons in the hidden layer...")
    W1, W2, loss_history, = back_propagation(X_train, d1_train, d2_train, l, epochs=epochs)
    losses_by_neurons[l] = loss_history

    predictions = predict(X_test, W1, W2)
    predicted_d1 = predictions[:, 0]
    # predicted_d2 = predictions[:, 1].round()
    predicted_d2 = predictions[:, 1]
    to_show = test_data.copy()

    for i in range(4):
        to_show[columns[i]] = denorm_array(to_show[columns[i]], np.min(data[columns[i]]), np.max(data[columns[i]]))
    predicted_d1_show = denorm_array(predicted_d1, np.min(data[columns[3]]), np.max(data[columns[3]]))

    results_df = pd.DataFrame({
        'x1': to_show['x1'],
        'x2': to_show['x2'],
        'x3': to_show['x3'],
        'True d1': to_show['d1'],
        'Predicted d1': predicted_d1_show,
        'True d2': to_show['d2'],
        'Predicted d2': predicted_d2
    })
    print(f'Результат навчання на {l} нейронах у прихованому шарі')
    print(results_df)
    print(f'MAE: {np.sum(np.abs(predicted_d1_show - to_show["d1"])) / len(test_data)}')
    result_data['W1'].append(W1)
    result_data['W2'].append(W2)
    result_data['LOSS'].append(loss_history)

plt.figure(figsize=(10, 6))
for l, losses in losses_by_neurons.items():
    plt.plot(range(epochs), losses, label=f'{l} neurons')

plt.title('Залежність помилки від кількості нейронів у прихованому шарі')
plt.xlabel('Ітерації')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()
