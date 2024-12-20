

if __name__ == "__main__":
    b = 207
    r = -1
    d = 0
    while r != 1:
        d = int(a / b)
        r = a - d * b
        print(f"{a} = {b}*{d} + {r}")
        a = b
        b = r