import numpy as np


class GF:
    def __init__(self, a, n=16, p="10011"):
        self.n = n
        self.p = p
        self.decimals = int(np.log2(self.n))
        if type(a) == int:
            self.a = a
        elif type(a) == str:
            self.a = int(a, 2)

    def __str__(self):
        return bin(self.a)[2:]

    def str_with_zeros(self):
        res = bin(self.a)[2:]
        while len(res) < self.decimals:
            res = "0" + res
        return res

    def copy(self):
        res = GF(self.a, self.n)
        res.p = self.p
        return res

    def reduce(self):
        res = self.__str__()
        # print(f"p={self.p}, dec={self.decimals}")
        while len(res.__str__()) > self.decimals:
            # print("res:", res)
            res = self.add_str(res, self.p + "0" * (len(res) - len(self.p)))
            while res[0] == "0":
                res = res[1:]
        return GF(int(res, 2))

    def add_str(self, one, other):
        a = "".join(list(reversed(one)))
        b = "".join(list(reversed(other)))
        if len(a) < len(b):
            a, b = b, a
        # print(f"ADDING {a} + {b} :", end=" ")
        for i in range(len(b)):
            ai = int(a[i])
            bi = int(b[i])
            a = a[:i] + str((ai + bi) % 2) + a[i + 1:]
        # print(a)
        return "".join(list(reversed(a)))

    def __add__(self, other):
        a = "".join(list(reversed(self.__str__())))
        if type(other) == GF:
            b = "".join(list(reversed(other.__str__())))
        if type(other) == str:
            b = "".join(list(reversed(other)))
        if len(a) < len(b):
            a, b = b, a
        # print(f"ADDING {a} + {b} :", end=" ")
        for i in range(len(b)):
            ai = int(a[i])
            bi = int(b[i])
            a = a[:i] + str((ai + bi) % 2) + a[i + 1:]
        # print(a)
        return GF(int("".join(list(reversed(a))), 2))

    def __mul__(self, other):
        a = self.__str__()
        b = other.__str__()
        res = "0"
        for i in range(len(b)):
            if b[-i - 1] == '1':
                # print(f"{res} + {self.__str__() + "0"*i} =", end=" ")
                res = self.add_str(res, str(self.__str__() + "0" * i))
                # print(res)
        # print("MULT:", res)
        res = GF(int(res, 2))
        res = res.reduce()
        # print("Reduced:", res)
        return res

    def __truediv__(self, other):
        for i in range(self.n):
            if GF(i, self.n) * other == self:
                return GF(i, self.n)
        return None

    def __eq__(self, other):
        if other is None:
            return False
        return self.a == other.a


def z2_z_x_part():
    z = list(i for i in range(8))
    z2 = list(i for i in range(8))
    x = list(i for i in range(8))
    x2 = list(i for i in range(8))
    for i in range(8):
        x[i] = GF(x[i])
        z[i] = GF(z[i])
        z2[i] = z[i] * z[i] + z[i]
        div = (GF(1) / (z[i] * z[i]))
        if div is None:
            x2[i] = None
        else:
            # right part of ex
            x2[i] = x[i] + GF(1) + GF("10") * div
        print(f"{z[i]}\t{z2[i]}\t{x[i]}\t{div}\t{x2[i]}")

    x_z = list()
    for i in range(8):
        for j in range(8):
            if z2[i] == x2[j]:
                x_z.append((x[j], z[i]))

    print("(x,z):")
    for pair in x_z:
        print(f"({pair[0]}, {pair[1]})", end=", ")
    print()
    print("(x,y):")
    for pair in x_z:
        print(f"({pair[0]}, {pair[0] * pair[1]})", end=", ")


def dots_sum():
    def division(a, b, p):
        for j in range(p):
            if b * j % p == a:
                return j

    x1 = 17
    y1 = 9
    x2 = 4
    y2 = 28
    p = 31
    i = 2
    while i < 50:
        i += 1
        print(f"m = ({y2} - {y1})/({x2} - {x1})", end="")
        m = division((y2 - y1) % p, (x2 - x1) % p, p)
        print(f" = {m}")
        x3 = (m ** 2 - x1 - x2) % p
        print(f"x3 = ({m} ** 2 - {x1} - {x2}) = {x3}")
        y3 = (m * (x1 - x3) - y1) % p
        print(f"y3 = ({m} * ({x1} - {x3}) - {y1}) = {y3}")
        print(f"{i}P = ({x1},{y1}) + ({x2},{y2}) = ({x3},{y3})")
        if (x3, y3) == (x1, y1):
            print("END")
            break
        x2 = x3
        y2 = y3


if __name__ == '__main__':
    z = list()
    z2 = list()
    x = [GF(1), GF("1000")]
    x2 = [GF(0), GF("11")]
    for i in range(16):
        z.append(GF(i))
        z2.append(GF(i)*GF(i) + GF(i))

    x_z = list()
    for i in range(len(z)):
        for j in range(len(x)):
            if z2[i] == x2[j]:
                x_z.append((x[j], z[i]))

    print("(x,z):")
    for pair in x_z:
        print(f"({pair[0]}, {pair[1]})", end=", ")
    print()
    print("(x,y):")
    for pair in x_z:
        print(f"({pair[0]}, {pair[0] * pair[1]})", end=", ")
