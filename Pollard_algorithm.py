import math
from prettytable import PrettyTable


def f(x):
    return x ** 2 + 1


table = PrettyTable(["i", "a", "b", "d"])
# n = 98587
n = 3714169
a = 1 % n
b = f(a) % n
print("n =", n)
for i in range(100):
    d = math.gcd(abs(a - b), n)
    # table.add_row([i, a, b, d])
    print(f"a({i})={a}, b({i})={b}, d={d}")
    a = f(a) % n
    b = f(f(b) % n) % n
    if 1 < d < n:
        print("END")
        break
# print(table)
