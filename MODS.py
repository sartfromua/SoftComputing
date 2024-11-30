import numpy as np
from matplotlib import pyplot as plt
from Evolution_modeling import Population


def function1d(x):
    y = 100 * np.sqrt(100 - x**2) * np.cos(x**2) * np.cos(x) / ((x**2+10) * np.log(100-x**2))
    return 5 + x * np.cos(x)-y


def Rastrigin_function(x):
    x, y = x
    return 20 + round(x**2 - 10*np.cos(2*np.pi*x)) + round(y**2 - 10*np.cos(2*np.pi*y))


def Ackley_function(x):
    x, y = x
    return (-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) -
            np.exp(0.5 * (np.cos(2 * np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20)


def Ackley_function2(x, y):

    return (-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) -
            np.exp(0.5 * (np.cos(2 * np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20)



def Himmelblau_function(x):
    x, y = x
    return (x**2 + y - 11)**2 + (x+y**2 - 7)**2

def Himmelblau_function2(x, y):
    return (x**2 + y - 11)**2 + (x+y**2 - 7)**2


class MODS:
    def __init__(self, dim, function, a=-5, b=5, size=20):
        self.size = size
        self.function = function
        self.dim = dim
        self.a = a
        self.b = b
        self.solutions = np.random.uniform(low=a, high=b, size=size * dim)
        if dim > 1:
            self.solutions = np.reshape(self.solutions, shape=(size, dim))

        self.values = np.array([function(x) for x in self.solutions])

    def learn(self):
        if self.dim == 1:
            self.learn1d()
        elif self.dim == 2:
            self.learn2d()

    def learn1d(self, sigma=1):
        random_population = self.solutions + np.random.normal(0, sigma, self.solutions.shape)
        for x in random_population:
            if x < self.a:
                x += self.b - self.a
            if x > self.b:
                x -= self.b - self.a
        mean_population = list()
        for _ in range(self.size):
            mean_population.append(np.mean(np.random.choice(self.solutions, size=2, replace=False)))

        solutions = np.append(self.solutions, random_population)
        solutions = np.append(solutions, mean_population)
        values = np.array([self.function(x) for x in solutions])

        solutions = zip(solutions, values)
        solutions = np.array(sorted(solutions, key=lambda x: x[1]))
        self.solutions = solutions[:, 0][:self.size]
        self.values = solutions[:, 1][:self.size]

    def return_point_into_rect(self, x):
        for i in range(x.shape[0]):
            if x[i] > self.b:
                x[i] -= self.b - self.a
            if x[i] < self.a:
                x[i] += self.b - self.a
        return x

    def get_two_points(self):
        i = np.random.randint(0, self.size)
        j = i
        while j == i:
            j = np.random.randint(0, self.size)
        x = self.solutions[i]
        y = self.solutions[j]
        return x, y

    def learn2d(self):
        transferred_solutions = list()
        for _ in range(int(self.size/2)):
            x, y = self.get_two_points()
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, self.b-self.a)
            new_x = np.array([x[0] + distance*np.cos(angle), x[1] + distance*np.sin(angle)])
            new_y = np.array([y[0] + distance*np.cos(angle), y[1] + distance*np.sin(angle)])

            new_x = self.return_point_into_rect(new_x)
            new_y = self.return_point_into_rect(new_y)

            transferred_solutions.append(new_x)
            transferred_solutions.append(new_y)
        transferred_solutions = np.array(transferred_solutions)

        rotated_solutions = list()
        for _ in range(int(self.size/2)):
            x, y = self.get_two_points()
            if self.function(x) > self.function(y):
                x, y = y, x

            angle = np.random.uniform(0, 2 * np.pi)
            y[0] = y[0] + (y[0] - x[0]) * np.cos(angle) - (y[1]-x[1]) * np.sin(angle)
            y[1] = y[1] + (y[0] - x[0]) * np.sin(angle) - (y[1] - x[1]) * np.cos(angle)
            y = self.return_point_into_rect(y)
            rotated_solutions.append(x)
            rotated_solutions.append(y)
        rotated_solutions = np.array(rotated_solutions)

        compressed_solutions = list()
        for _ in range(int(self.size/2)):
            x, y = self.get_two_points()
            if self.function(x) > self.function(y):
                x, y = y, x

            k = 2
            y[0] = (y[0] + x[0]) / k
            y[1] = (y[1] + x[1]) / k
            y = self.return_point_into_rect(y)
            compressed_solutions.append(x)
            compressed_solutions.append(y)
        compressed_solutions = np.array(compressed_solutions)

        solutions = np.append(self.solutions, transferred_solutions)
        solutions = np.append(solutions, rotated_solutions)
        solutions = np.append(solutions, compressed_solutions)
        solutions = np.reshape(solutions, (self.size*4, 2))
        values = np.array([self.function(x) for x in solutions])

        solutions = list(zip(solutions, values))
        solutions = list((sorted(solutions, key=lambda x: x[1])))
        self.solutions = list()
        self.values = list()
        for x, y in solutions[:self.size]:
            self.solutions.append(x)
            self.values.append(y)
        self.solutions = np.array(self.solutions)
        self.solutions = np.reshape(self.solutions, (self.size, 2))
        self.values = np.array(self.values)


if __name__ == "__main__":
    function = Ackley_function

    model = MODS(2, function, size=20)
    iterations = 50
    results = list()
    for i in range(iterations):
        model.learn()
        results.append(model.values[0])
    print("Solutions:", *np.round(model.solutions, 4)[:6])
    print("Values:", np.round(model.values, 4)[:6])
    plt.plot(results, label="MODS")

    population = Population(Ackley_function2, a=-5, b=5)
    results = list()
    for i in range(iterations):
        population.next_generation("ES", "SE", "M2", "OD")
        results.append(population.get_best_individual().adaptation)
    plt.plot(results, label="Evolution")
    print(population.get_best_individual())
    plt.legend()
    plt.show()
