import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time
from prettytable import PrettyTable

coords = {
    "Київ": (-27, 147),
    "Чернігів": (43, 213),
    "Суми": (148, 193),
    "Полтава": (134, 100),
    "Харків": (270, 91),
    "Дніпро": (191, -0),
    "Кропивницький": (52, 4),
    "Черкаси": (12, 64),
    "Одеса": (-38, -105),
    "Симферопіль": (150, -230),
    "Херсон": (128, -127),
    "Запоріжжя": (230, -82),
    "Вінниця": (-128, 36),
    "Житомир": (-133, 161),
    "Рівне": (-224, 195),
    "Луцьк": (-303, 203),
    "Львів": (-348, 99),
    "Хмельницький": (-206, 90),
    "Тернопіль": (-270, 70),
    "Чернівці": (-245, -2),
    "Івано-Франківськ": (-319, 4),
    "Ужгород": (-389, 4),
    "Донецьк": (332, -34),
    "Луганськ": (389, 45),
    "Миколаїв": (35, -79)
}


def draw(path, distance):
    name = "map.jpg"
    img = mpimg.imread(name)

    plt.imshow(img)
    for i in range(len(path)-1):
        x = [500+coords[path[i]][0], 500+coords[path[i+1]][0]]
        y = [375-coords[path[i]][1], 375-coords[path[i+1]][1]]
        plt.plot(x, y, "k")
    x = [500 + coords[path[0]][0], 500 + coords[path[-1]][0]]
    y = [375 - coords[path[0]][1], 375 - coords[path[-1]][1]]
    plt.plot(x, y, "k")
    plt.grid(visible=False)
    plt.savefig(f'{int(distance)}.png')
    plt.show()


def read_distances(filename):
    distances = dict()
    with open(filename) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("\n", "").replace(" ", "")
            # print(f"|{lines[i]}|")
        names = lines[0][1:].split(";")
        for line in lines[1:]:
            items = line.split(";")
            name = items[0]
            items = items[1:]
            if name == "Львов": name = "Львів"
            if name == "Чернігів\n": name = "Чернігів"
            for i in range(len(items)):
                if items[i] == '' or items[i] == '\n':
                    continue
                items[i] = items[i]
                if name not in distances.keys():
                    distances[name] = dict()
                distances[name][names[i]] = float(items[i])
    return distances


def get_dict_copy(dict1: dict):
    res = dict()
    for key in dict1.keys():
        for item in dict1[key]:
            if key not in res.keys():
                res[key] = dict()
            res[key][item] = dict1[key][item]
    return res


class Graph:
    def __init__(self, distances):
        self.distances = get_dict_copy(distances)
        self.pheromones = get_dict_copy(distances)
        self.make_pheromones()
        self.path = []
        self.cities_left = []
        self.reset_path()
        self.best_path = None

    def __str__(self):
        res = ""
        for key in self.distances.keys():
            res += str(key) + ": \t" + str(self.distances[key]) + "\n"
        return res

    def print_pheromones(self):
        table = PrettyTable(["\\", *list(self.pheromones.keys())])
        table.float_format = "0.2"
        names = list(self.pheromones.keys())
        for i in range(len(names)):
            vals = [names[i]]
            for j in range(len(names)):
                if names[j] in self.pheromones[names[i]].keys():
                    vals.append(self.pheromones[names[i]][names[j]])
                else:
                    vals.append(0)
            table.add_row(vals)
        print(table)

    def update_pheromones(self, paths):
        for key in self.pheromones.keys():
            for key1 in self.pheromones[key].keys():
                self.pheromones[key][key1] *= 0.99
        for path in paths:
            length = self.get_path_length(path)
            r = np.pow(length, -2)
            for i in range(len(path)-1):
                self.pheromones[path[i]][path[i+1]] = self.pheromones[path[i]][path[i+1]] + (1+r)*np.pow(self.F(path), 1/10)
                # self.pheromones[path[i]][path[i+1]] *= 1.05

    def F(self, path):
        koef = 5
        res = 0
        for i in range(len(path)-1):
            res += koef/self.distances[path[i]][path[i+1]]
        return res

    def make_pheromones(self):
        for key in self.pheromones.keys():
            for key1 in self.pheromones[key].keys():
                self.pheromones[key][key1] = np.random.uniform(0, 1)
                # self.pheromones[key][key1] = 1

    def get_path_length(self, path):
        length = 0
        for i in range(len(path)-2):
            length += self.distances[path[i]][path[i+1]]
        length += self.distances[path[-1]][path[0]]
        return length

    def print_best_path(self):
        for city in self.best_path[0]:
            print(city, end=" -> ")
        print("Київ")

    def reset_path(self):
        self.path = ["Київ"]
        self.cities_left = list(self.distances.keys())
        self.cities_left.remove("Київ")

    def find_path(self, ants=20):
        paths = []
        for i in range(ants):
            self.reset_path()
            while len(self.path) != len(self.distances.keys()):
                self.next_city()
            length = self.get_path_length(self.path)
            if self.best_path is not None:
                if length < self.best_path[1]:
                    self.best_path = (self.path, length)
            else:
                self.best_path = (self.path, self.get_path_length(self.path))
            paths.append(self.path)
        self.update_pheromones(paths)

    def next_city(self):
        ways = self.pheromones[self.path[-1]]
        info = self.distances[self.path[-1]]
        chances = dict()
        k = 1
        h = 2
        alpha = 1
        beta = 2
        for city in self.cities_left:
            chances[city] = ways[city]
        summa = 0
        for city in self.cities_left:
            try:
                summa += np.pow(ways[city], alpha) * np.pow(info[city], -beta)
            except Exception as e:
                print(ways[city], k, h)
                assert OverflowError
        for city in self.cities_left:
            chances[city] = np.pow(ways[city], alpha) * np.pow(info[city], -beta) / summa

        chance_limit = np.random.uniform(0, 1)
        chance = 0
        for city in self.cities_left:
            chance += chances[city]
            if chance > chance_limit:
                self.path.append(city)
                self.cities_left.remove(city)
                break


if __name__ == "__main__":
    distances = read_distances("city_distances.csv")
    best_path = ("", 100000)
    start = time.time()
    tries = 50
    print("Processing...\n[", end="")
    for i in range(tries):
        graph = Graph(distances)
        for j in range(10):
            graph.find_path(ants=20)
        graph.find_path()
        path = graph.path
        distance = graph.best_path[1]
        if distance < best_path[1]:
            best_path = graph.best_path
        print(f"\r[{"|"*int(100*i/tries)}{" "*int(100-100*i/tries)}]", flush=True, end="")
    print()
    path = best_path[0]
    draw(*best_path)
    print(path)
    print("Distance:", best_path[1])
    print("time passed:", time.time() - start)

