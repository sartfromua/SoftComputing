import numpy as np
from matplotlib import pyplot as plt

data = [[1, 32, 3, 3.141462],
        [2, 32, 2, 6.29715],
        [3, 45, 3, 0.316198],
        [4, 65, 4, 2.229654],
        [5, 3, 53, 12.56472],
        [6, 2, 23, 0.973952],
        [7, 1, 12, 1.339259],
        [8, 5, 6, 46.39461],
        [9, 4, 5, 1.551375],
        [5, 3, 7, 12.8396],
        [4, 7, 6, 1.986992],
        [3, 5, 4, 0.673534],
        [2, 2, 11, 5.947558],
        [1, 8, 21, 3.146682],
        [2, 2, 21, 5.64757],
        [6, 11, 76, 0.405182],
        [7, 24, 34, 1.219278],
        [8, 56, 54, 47.27631],
        [9, 75, 23, 1.770304],
        [23, 43, 76, 3.150859],
        [43, 32, 32, 3.245165],
        [23, 21, 16, 2.905126],
        [54, 43, 38, 0.849992],
        [34, 21, 43, 1.380608],
        ]


def m_function(x, trapeze):
    # for trapeze function
    # m_left, m_right, alpha, beta = trapeze
    # h = 1
    # if m_left <= x <= m_right:
    #     return h
    # if m_left - alpha <= x <= m_left:
    #     tg = h / alpha
    #     return (x - (m_left - alpha)) * tg
    # if m_right <= x <= m_right + beta:
    #     tg = h / beta
    #     return (beta - (x - m_right)) * tg
    # return 0
    # Gaussian function
    c, sigma = trapeze
    return np.exp(-(x - c) ** 2 / (sigma ** 2))


def temperature(t):
    return 10**3/(10**3+t)


def Cauchy_distribution(x, iteration):
    t = temperature(iteration)
    return t / (t**2 + x**2)


class ANFIS:
    def __init__(self, input_size=3):
        self.parameters = 3
        self.input_size = input_size
        self.function_parameters = np.random.uniform(size=input_size*self.input_size*self.parameters)
        # A1 B1 C1 a1 b1 c1
        # A2 B2 C2 a2 b2 c2
        # A3 B3 C3 a3 b3 c3
        self.input_size = input_size

    def predict(self, inputs):
        inputs = np.array(inputs)
        alphas = np.zeros(self.input_size)
        self.function_parameters = np.reshape(self.function_parameters,
                                              (self.input_size, self.input_size * self.parameters))
        for i in range(self.input_size):
            alphas[i] = (m_function(inputs[0], [self.function_parameters[i][0], self.function_parameters[i][1]]) *
                         m_function(inputs[1], [self.function_parameters[i][2], self.function_parameters[i][3]]) *
                         m_function(inputs[2], [self.function_parameters[i][4], self.function_parameters[i][5]]))
        # print(alphas)
        if sum(alphas) == 0:
            norm_alphas = alphas
        else:
            norm_alphas = alphas / np.sum(alphas)
        res = 0
        for i in range(self.input_size):
            res += norm_alphas[i] * (inputs[0]*self.function_parameters[i][6] +
                                     inputs[1]*self.function_parameters[i][7] +
                                     inputs[2]*self.function_parameters[i][8])
        self.function_parameters = np.reshape(self.function_parameters,
                                              (self.input_size * self.input_size * self.parameters))
        return res

    def stochastic_learning(self, X, y, learning_rate=0.01):
        t = 0
        error = 100
        epsilon = 10
        losses = list()
        print("X.shape:", X.shape)
        print("self.function_parameters.shape:", self.function_parameters.shape)
        print(self.function_parameters)
        while error > epsilon:
            i = t % X.shape[0]
            x = X[i]
            y_before = self.predict(x)
            j = np.random.randint(self.function_parameters.shape[0])
            zeta = np.random.uniform(low=-np.pi/2, high=np.pi/2)
            # delta = learning_rate * temperature(t) * np.tan(zeta)
            delta = learning_rate * zeta
            self.function_parameters[j] += delta
            y_after = self.predict(x)
            if abs(y[i] - y_before) < abs(y[i] - y_after):
                chance = np.random.uniform()
                if chance > temperature(t):
                    self.function_parameters[j] -= delta

            t += 1
            if t % X.shape[0] == 0:
                error = 0
                for i in range(X.shape[0]):
                    error += abs(self.predict(X[i]) - y[i])

                losses.append(error)
                if t / X.shape[0] % 1000 == 0:
                    print()
                    print(self.function_parameters)
                    print(np.round(np.array(y[:10]), 2), sep="\t")
                    print(np.round(np.array([self.predict(x) for x in X[:10]]), 2), sep="\t")
                    print("Error:", error)
                    plt.plot(losses)
                    plt.show()
        plt.plot(losses)
        plt.show()


if __name__ == '__main__':
    x = list()
    y = list()
    for numbers in data:
        x.append(numbers[:3])
        y.append(numbers[3])
    x = np.array(x)
    x = np.reshape(x, (len(data), 3))
    y = np.array(y)
    network = ANFIS()
    network.stochastic_learning(x, y)
    for i in range(x.size[0]):
        print(x[i], y[i], network.predict(x[i]))

