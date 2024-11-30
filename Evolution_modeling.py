import numpy as np
from prettytable import PrettyTable
from time import time
from matplotlib import pyplot as plt

p = 0


def denorm_phen(phen, a, epsilon):
    return phen*epsilon + a


def to_phen(x: str) -> int:
    return int(x, 2)


def to_gen(x: int) -> str:
    global p
    res = bin(x)[2:]
    return "0"*max(p-len(res), 0) + res


def hemming_distance(x: str, y: str) -> int:
    res = 0
    x = x[0] + x[1]
    y = y[0] + y[1]
    for i in range(len(x)):
        if x[i] != y[i]:
            res += 1
    return res


def Rastrigin_function(x, y):
    return 20 + int(x**2 - 10*np.cos(2*np.pi*x)) + int(y**2 - 10*np.cos(2*np.pi*y))


def Ackley_function(x, y):
    return (-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) -
            np.exp(0.5 * (np.cos(2 * np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20)


def Himmelblau_function(x, y):
    return (x**2 + y - 11)**2 + (x+y**2 - 7)**2


class Individual:
    def __init__(self, phen, fitness_function, epsilon=0.1, a=-5):
        self.genes = np.array([to_gen(phen[0]), to_gen(phen[1])])
        self.phen = phen
        self.epsilon = epsilon
        self.a = a
        self.fitness_function = fitness_function
        self.adaptation = 0
        self.count_adaptation()

    def count_adaptation(self):
        self.adaptation = self.fitness_function(denorm_phen(self.phen[0], self.a, self.epsilon),
                                                denorm_phen(self.phen[1], self.a, self.epsilon))

    def __str__(self):
        return (f"({denorm_phen(self.phen[0], self.a, self.epsilon):.3f}, {denorm_phen(self.phen[1], self.a, self.epsilon):.3f}):"
                f" {self.adaptation:.3f}")


# Как проходят операции с генами у родителей? попарно 1 и 2 ген у каждого родителя?


class Population:
    def __init__(self, fitness_function, a=-5, b=5, epsilon=0.001, n=30):
        self.n = n
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.k = int((self.b - self.a) / epsilon + 1)
        global p
        p = int(np.log2(self.k)) + 1
        self.k = 2**p
        self.epsilon = abs((self.a - self.b) / self.k)
        self.fitness_function = fitness_function
        self.individuals = []
        # print(f"k={self.k}, a={self.a}, b={self.b}, epsilon={self.epsilon}, p={p}")
        for i in range(n):
            self.individuals.append(Individual(np.array(np.random.choice(range(self.k), size=2)), fitness_function, self.epsilon, self.a))
        self.adaptation = sum([ind.adaptation for ind in self.individuals])
        self.generation = 0

    def __str__(self):
        res = "\n"
        i = 0
        for ind in sorted(self.individuals, key=lambda ind: ind.adaptation):
            res += ind.__str__() + "  \t"
            i += 1
            if i%5 == 0:
                res += "\n"
        res += f"\nAdaptation: {self.adaptation:.3f}"
        return res

    def get_average_distance(self):
        # distance = 0
        # for i in range(len(self.individuals)-1):
        #     distance += hemming_distance(self.individuals[i].genes, self.individuals[i+1].genes)
        distance = 0
        for ind1 in self.individuals:
            for ind2 in self.individuals:
                distance = max(distance, hemming_distance(ind1.genes, ind2.genes))
        return distance/len(self.individuals)

    def show_adaptation(self):
        print(f"Generation {self.generation}: {self.adaptation:.5f}")

    def get_adaptation(self):
        return self.adaptation

    def get_best_individual(self):
        return sorted(self.individuals, key=lambda x: x.adaptation)[0]

    @staticmethod
    def crossover(gen1, gen2, mode):
        gen_new = []
        # One dot
        if mode == "OD":
            for i in range(len(gen1)):
                k = np.random.choice(range(len(gen1[i]))[1:-1])
                gen_new.append(gen1[i][:k] + gen2[i][k:])
        # Two dots
        if mode == "TD":
            for i in range(len(gen1)):
                k1 = np.random.choice(range(len(gen1[i]))[1:-2])
                k2 = np.random.choice(range(k1, len(gen1[i]))[1:-1])
                gen_new.append(gen1[i][:k1] + gen2[i][k1:k2] + gen1[i][k2:])
        return gen_new

    @staticmethod
    def mutation(gen, mode="M1"):
        chance = 0
        if mode == "M1":
            chance = 0.001
        if mode == "M2":
            chance = 0.05
        if mode == "M3":
            chance = 0.1
        genes_new = []
        for i in range(len(gen)):
            k = np.random.choice(range(len(gen[i]) - 1))
            s = gen[i][k]
            if np.random.random() < chance:
                s = "0" if s == "1" else "1"
            genes_new.append(gen[i][:k] + s + gen[i][k+1:])
        return genes_new

    def get_parents(self, mode):
        pair = [None, None]
        # Panmixia
        if mode == "PM":
            pair = np.random.choice(self.individuals, size=2)
        # Outbreeding
        if mode == "OB":
            pair[0] = np.random.choice(self.individuals)
            individuals = self.individuals.copy()
            individuals.remove(pair[0])
            hem_dist = list()
            for individual in individuals:
                hem_dist.append((hemming_distance(pair[0].genes, individual.genes), individual))
            hem_dist = sorted(hem_dist, key=lambda x: x[0])
            pair[1] = hem_dist[0][1]
        # Inbreeding
        if mode == "IB":
            pair[0] = np.random.choice(self.individuals)
            individuals = self.individuals.copy()
            individuals.remove(pair[0])
            hem_dist = list()
            for individual in individuals:
                hem_dist.append((hemming_distance(pair[0].genes, individual.genes), individual))
            hem_dist = sorted(hem_dist, key=lambda x: x[0])
            pair[1] = hem_dist[-1][1]
        # Selective
        if mode == "SE":
            individuals = sorted(self.individuals, key=lambda x: x.adaptation)
            pair = np.random.choice(individuals[:len(individuals)//2], size=2)

        if pair[0] == pair[1]:
            pair = self.get_parents(mode)
        return tuple(pair)

    def next_generation(self, selection_mode, parents_mode, mutation_mode, crossover_mode):
        self.generation += 1
        children = []
        parents = self.individuals.copy()
        while len(children) != len(parents):
            pair = self.get_parents(parents_mode)
            genes1 = pair[0].genes
            genes2 = pair[1].genes
            genes_new = self.crossover(genes1, genes2, crossover_mode)
            genes_new = self.mutation(genes_new, mode=mutation_mode)
            child = Individual([to_phen(genes_new[0]), to_phen(genes_new[1])], self.fitness_function, self.epsilon, self.a)
            # child.genes = genes_new
            # child.phen = np.array(to_phen(genes_new[0]))
            # print(genes_new)
            child.count_adaptation()
            sd_double = False
            if selection_mode == "SD":
            # if True:
                for ch in children:
                    if ch.genes[0] == child.genes[0] and ch.genes[1] == child.genes[1]:
                        sd_double = True
                        break
            if not sd_double:
                children.append(child)

        # Conventional selection
        if selection_mode == "CS":
            self.individuals = children

        # Selection with displacement
        if selection_mode == "SD":
            self.individuals = children

        # Elite selection
        if selection_mode == "ES":
            children.extend(parents)
            self.individuals = sorted(children, key=lambda x: x.adaptation)[:len(parents)]

        self.adaptation = sum([ind.adaptation for ind in self.individuals])


def print_mutation_table(function, default_params, mutation_mods, iterations=300):
    table_mutation = PrettyTable(["Mutation", "Result", "Time"])
    for mutation_mode in mutation_mods:
        time_start = time()
        population = Population(function, a=-5, b=5)
        adaptation = list()
        params = default_params.copy()
        params[2] = mutation_mode
        # while population.get_average_distance() > 3/population.n:
        for i in range(iterations):
            population.next_generation(*params)
            adaptation.append(population.get_adaptation())
        plt.plot(adaptation, label=f"{mutation_mode} mutation")
        plt.legend()
        table_mutation.add_row([mutation_mode, population.get_best_individual(), time() - time_start])
    plt.show()
    print(f"Parameters: {default_params}")
    print(table_mutation)


def print_parents_table(function, default_params, parents_mods, iterations=300):
    table_parents = PrettyTable(["Parents mode", "Result", "Time"])
    for parents_mode in parents_mods:
        time_start = time()
        population = Population(function, a=-5, b=5)
        adaptation = list()
        params = default_params.copy()
        params[1] = parents_mode
        for i in range(iterations):
            population.next_generation(*params)
            adaptation.append(population.get_adaptation())
        plt.plot(adaptation, label=f"{parents_mode} parents choosing mode")
        plt.legend()
        table_parents.add_row([parents_mode, population.get_best_individual(), time() - time_start])
    plt.show()
    print(f"Parameters: {default_params}")
    print(table_parents)


def print_crossover_table(function, default_params, crossover_mods, iterations=300):
    table_crossover = PrettyTable(["Crossover", "Result", "Time"])
    for crossover_mode in crossover_mods:
        time_start = time()
        population = Population(function, a=-5, b=5)
        adaptation = list()
        params = default_params.copy()
        params[3] = crossover_mode
        for i in range(iterations):
            population.next_generation(*params)
            adaptation.append(population.get_adaptation())
        plt.plot(adaptation, label=f"{crossover_mode} crossover")
        plt.legend()
        table_crossover.add_row([crossover_mode, population.get_best_individual(), time() - time_start])
    plt.show()
    print(f"Parameters: {default_params}")
    print(table_crossover)


def print_selection_table(function, default_params, selection_mods, iterations=300):
    table_selection = PrettyTable(["Selection mode", "Result", "Time"])
    for selection_mode in selection_mods:
        time_start = time()
        population = Population(function, a=-5, b=5)
        adaptation = list()
        params = default_params.copy()
        params[0] = selection_mode
        for i in range(iterations):
            population.next_generation(*params)
            adaptation.append(population.get_adaptation())
        plt.plot(adaptation, label=f"{selection_mode} selection")
        plt.legend()
        table_selection.add_row([selection_mode, population.get_best_individual(), time() - time_start])
    plt.show()
    print(f"Parameters: {default_params}")
    print(table_selection)


def print_stop_table(function, params, iterations):
    table_stop = PrettyTable(["Stop mode", "Result", "Time"])
    time_start = time()
    population = Population(function, a=-5, b=5)
    adaptation = list()
    for i in range(iterations):
        population.next_generation(*params)
        adaptation.append(population.get_adaptation())
    plt.plot(adaptation, label="Iteration number curve")

    population.show_adaptation()
    # print(population)
    table_stop.add_row(["Iteration number", population.get_best_individual(), time() - time_start])

    time_start = time()
    population = Population(function, a=-5, b=5)
    adaptation = list()
    while population.get_average_distance() > 3/population.n:
        population.next_generation(*params)
        # population.show_adaptation()
        adaptation.append(population.get_adaptation())
        # print(population.get_average_distance())
        # if population.generation % 250 == 0 and population.generation > 0:
        #     plt.title("Individuals distance curve")
        #     plt.plot(adaptation)
        #     plt.show()
    plt.plot(adaptation, label="Individuals distance curve")
    population.show_adaptation()
    # print(population)
    table_stop.add_row(["Individuals distance", population.get_best_individual(), time() - time_start])

    time_start = time()
    population1 = Population(function, a=-5, b=5)
    population2 = Population(function, a=-5, b=5)
    adapt1 = population1.get_adaptation()
    adapt2 = population2.get_adaptation()
    adaptation1 = list()
    adaptation2 = list()
    while abs(adapt1 - adapt2) / population.n > 0.01:
        population1.next_generation(*params)
        population2.next_generation(*params)
        # population.show_adaptation()
        adapt1 = population1.get_adaptation()
        adapt2 = population2.get_adaptation()
        adaptation1.append(adapt1)
        adaptation2.append(adapt2)
        # if population1.generation % 250 == 0 and population1.generation > 0:
        #     plt.title("Populations distance curve")
        #     plt.plot(adaptation1)
        #     plt.plot(adaptation2)
        #     plt.show()
    plt.plot(adaptation1, label="Population1 distance curve")
    plt.plot(adaptation2, label="Population2 distance curve")
    population1.show_adaptation()
    population2.show_adaptation()
    # print(population1)
    table_stop.add_row(["Populations distance", population.get_best_individual(), time() - time_start])

    plt.legend()
    plt.show()
    print(table_stop)


if __name__ == "__main__":
    # population = Population(Ackley_function, a=-5, b=5, epsilon=0.001)
    # print(population)
    # for i in range(100):
    #     population.next_generation("CS", "SE", "M2", "OD")
    #     population.show_adaptation()
    # print(population.get_best_individual())

    default_params = ["CS", "SE", "M2", "OD"]
    iterations = 300
    function = Himmelblau_function

    mutation_mods = ["M1", "M2", "M3"]
    print_mutation_table(function, default_params, mutation_mods, iterations)

    parents_mods = ["PM", "OB", "IB", "SE"]
    print_parents_table(function, default_params, parents_mods, iterations)

    crossover_mods = ["OD", "TD"]
    print_crossover_table(function, default_params, crossover_mods, iterations)

    selection_mods = ["CS", "SD", "ES"]
    print_selection_table(function, default_params, selection_mods, iterations)

    print_stop_table(function, default_params, iterations)

    print(f"Parameters: {default_params}")
