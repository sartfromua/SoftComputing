# Variant 2
import random
from prettytable import PrettyTable

r = 5


def prepare_values():
    x_values = [[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
    ]
    N = len(x_values)
    M = len(x_values[0])
    true_res = [int((not x[0] and x[1]) or (not x[2])) for x in x_values]
    threshold = 0.5
    res = list(-1 for i in range(len(x_values)))
    return x_values, true_res, threshold, res, N, M


def model():
    learning_rates = [0.2, 0.16, 0.12, 0.08, 0.04, 0.01]
    # learning_rates = [0.05, 0.1, 0.3]
    steps = [0 for i in learning_rates]
    k = 0
    # weights_values = [0.2, 0.1, 0.7]
    weights_values = [round(random.random(), 3) for i in range(3)]
    print("Weights:", weights_values)
    for learning_rate in learning_rates:
        x_values, true_res, threshold, res, N, M = prepare_values()
        weights = weights_values.copy()
        print(true_res)
        table = PrettyTable(
            ["w1", "w2", "w3", "teta", "x1", "x2", "x3", "a", "Y", "T", "eta*(T-Y)", "delta*w1", "delta*w2", "delta*w3"]
        )
        j = 0
        while res != true_res:
            x = x_values[j]
            activation = sum([x[i]*weights[i] for i in range(M)]) + 1
            res[j] = int(activation > threshold)
            if res[j] != true_res[j]:
                steps[k] += 1
                diff = true_res[j] - res[j]
                delta_values = [learning_rate*diff*x[i] for i in range(M)]
                table.add_row(list(round(el, r) for el in [*weights, threshold, *x, activation, res[j], true_res[j],
                                                           learning_rate * (true_res[j] - res[j]), *delta_values]))
                weights = [weights[i]+delta_values[i] for i in range(M)]
                j = 0
                continue
            table.add_row(list(round(el, r) for el in [*weights, threshold, *x, activation, res[j], true_res[j],
                   learning_rate*(true_res[j]-res[j]), *[0 for i in range(M)]]))
            j = 0 if j == 7 else j+1

            if steps[k] >= 500:
                break
        k += 1
        print("Learning rate:", learning_rate)
        print(table)

    lr_table = PrettyTable(["Learning rate", "Steps"])
    lr_table.add_rows([[learning_rates[i], steps[i]] for i in range(len(steps))])
    print(lr_table)


if __name__ == "__main__":
    model()

