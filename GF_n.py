import numpy as np


class GF:
    def __init__(self, a, n=8, p="1011"):
        self.n = n
        self.p = p
        self.decimals = int(np.log2(self.n))
        self.a = a

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
            res = self.add_str(res, self.p+"0"*(len(res) - len(self.p)))
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
            a = a[:i] + str((ai + bi) % 2) + a[i+1:]
        # print(a)
        return GF(int("".join(list(reversed(a))), 2))

    def __mul__(self, other):
        a = self.__str__()
        b = other.__str__()
        res = "0"
        for i in range(len(b)):
            if b[-i-1] == '1':
                # print(f"{res} + {self.__str__() + "0"*i} =", end=" ")
                res = self.add_str(res, str(self.__str__() + "0"*i))
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
        return self.a == other.a


if __name__ == '__main__':
    # z = list(0 for i in range(8))
    # z2_z = list(0 for i in range(8))
    # for i in range(8):
    #     z[i] = GF(i)
    #     div = (GF(1)/(z[i] * z[i]))
    #     if div is None:
    #         z2_z[i] = None
    #     else:
    #         z2_z[i] = z[i] + GF(1) + div * GF(int("10", 2))
    # print(*z)
    # print(*z2_z)

    a = [(1, 100), (1, 101), (10, 10), (10, 11), (11, 10), (11, 11), (110, 10), (110, 11), (111, 1)]
    for pair in a:
        print(f"({pair[0]}, {GF(int(str(pair[1]), 2))*GF(int(str(pair[0]), 2))})", end=", ")
