from matplotlib import pyplot as plt
import numpy as np


def draw_trapeze(trapeze):
    m_left, m_right, alpha, beta, h = trapeze
    plt.plot([m_left - alpha, m_left, m_right, m_right + beta], [0, h, h, 0])


def get_alpha_intersection(x, trapeze):
    m_left, m_right, alpha, beta, h = trapeze
    if m_left <= x <= m_right:
        return trapeze[4]
    if m_left - alpha <= x <= m_left:
        tg = h / alpha
        return (x - (m_left - alpha)) * tg
    if m_right <= x <= m_right + beta:
        tg = h / beta
        return (beta - (x - m_right)) * tg
    return -1


def get_trapeze_intersection(y, trapeze):
    m_left, m_right, alpha, beta, h = trapeze
    if y >= h:
        return -1
    tg = h / alpha
    new_m_left = m_left - alpha + y / tg
    new_alpha = new_m_left - (m_left - alpha)
    tg = h / beta
    new_m_right = m_right + beta - y / tg
    new_beta = (m_right + beta) - new_m_right
    return [new_m_left, new_m_right, new_alpha, new_beta, y]


def get_min_trapeze(alpha, trapeze):
    m_left, m_right, alpha, beta, h = trapeze


def draw_maximum(trapeze1, trapeze2):
    m_left1, m_right1, alpha1, beta1, h1 = trapeze1
    m_left2, m_right2, alpha2, beta2, h2 = trapeze2
    min_x = min(m_left1 - alpha1, m_left2 - alpha2)
    max_x = max(m_right1 + beta1, m_right2 + beta2)
    xs = np.linspace(min_x, max_x, num=1000)
    ys = np.array([max(get_alpha_intersection(x, trapeze1), get_alpha_intersection(x, trapeze2)) for x in xs])
    plt.plot(xs, ys)
    x_range = [min_x, max_x]
    z = -1
    for i in range(20):
        z = (x_range[1] - x_range[0]) / 2 + x_range[0]
        s1 = 0
        s2 = 0
        for i in range(len(xs)):
            if xs[i] > z:
                break
            s1 += ys[i]
        for j in range(i, len(xs)):
            s2 += ys[j]
        if s1 > s2:
            x_range = [x_range[0], z]
        else:
            x_range = [z, x_range[1]]
    plt.plot([z, z], [0, 1])
    plt.title("Result")
    return z


def exercise2():
    print("Ex2")
    x0 = 220
    y0 = 200
    A1 = [100, 200, 30, 40, 1]
    A2 = [200, 300, 20, 60, 1]
    B1 = [140, 240, 30, 40, 1]
    B2 = [240, 320, 50, 50, 1]
    C1 = [50, 100, 10, 30, 1]
    C2 = [100, 150, 20, 5, 1]
    alpha1 = min(get_alpha_intersection(x0, A1), get_alpha_intersection(y0, B1))
    alpha2 = min(get_alpha_intersection(x0, A2), get_alpha_intersection(y0, B2))
    print(f"alpha1 = {alpha1}\nalpha2 = {alpha2}")

    draw_trapeze(A1)
    draw_trapeze(B1)
    plt.plot([min(A1[0]-A1[2], B1[0]-B1[2]), max(A1[1]+A1[3], B1[1]+B1[3])], [alpha1, alpha1])
    plt.plot([x0, x0], [0, 1])
    plt.plot([y0, y0], [0, 1])
    plt.title("A1 and B1")
    plt.show()

    draw_trapeze(A2)
    draw_trapeze(B2)
    plt.plot([min(A2[0] - A2[2], B2[0] - B2[2]), max(A2[1]+A2[3], B2[1]+B2[3])], [alpha2, alpha2])
    plt.plot([x0, x0], [0, 1])
    plt.plot([y0, y0], [0, 1])
    plt.title("A2 and B2")
    plt.show()

    C1_new_trapeze = get_trapeze_intersection(alpha1, C1)
    C2_new_trapeze = get_trapeze_intersection(alpha2, C2)
    print(C1_new_trapeze)
    print(C2_new_trapeze)

    draw_trapeze(C1)
    plt.plot([C1[0]-C1[2], C1[1]+C1[3]], [alpha1, alpha1])
    plt.title("C1")
    plt.show()

    draw_trapeze(C2)
    plt.plot([C2[0]-C2[2], C2[1]+C2[3]], [alpha2, alpha2])
    plt.title("C2")
    plt.show()

    z = draw_maximum(C1_new_trapeze, C2_new_trapeze)
    print(f"Z = {z}")
    plt.show()


def xor_trapeze(trapeze1, trapeze2):
    m_left1, m_right1, alpha1, beta1, h1 = trapeze1
    m_left2, m_right2, alpha2, beta2, h2 = trapeze2
    h = min(h1, h2)
    alpha = h * (alpha1/h1 + alpha2/h2)
    m_left = m_left1+m_left2-alpha1-alpha2 + alpha
    beta = h * (beta1/h1 + beta2/h2)
    m_right = m_right1+m_right2+beta1+beta2 - beta
    return [m_left, m_right, alpha, beta, h]


def xor_list_trapezes(trapezes):
    res = xor_trapeze(trapezes[0], trapezes[1])
    for trapeze in trapezes[2:]:
        res = xor_trapeze(res, trapeze)
    return res


def exercise1():
    print("Ex1")
    A = [320, 320, 0, 0, 1]
    B = [310, 370, 40, 50, 1]
    C1 = [300, 300, 100, 0, 0.8]
    C2 = [0, 0, 0, 0, 0.2]
    E = [230, 240, 10, 20, 0.8]
    D1 = [310, 310, 0, 200, 0.2]
    D2 = [0, 0, 0, 0, 0.8]
    draw_trapeze(A)
    draw_trapeze(B)
    draw_trapeze(C1)
    draw_trapeze(E)
    draw_trapeze(D1)
    plt.show()
    ss = list()
    for C in [C1, C2]:
        for D in [D1, D2]:
            ss.append(xor_list_trapezes([A, B, C, D, E]))
    for s in ss:
        draw_trapeze(s)
    plt.show()
    min_x = 100000
    max_x = -10000
    for tup in ss:
        min_x = min(min_x, tup[0]-tup[2])
        max_x = max(max_x, tup[1]+tup[3])
    xs = np.linspace(min_x, max_x, num=10000)
    ys = np.zeros_like(xs)
    most_probably_x = list()
    low_probably_x = list()
    for i in range(len(xs)):
        for tup in ss:
            ys[i] = max(ys[i], get_alpha_intersection(xs[i], tup))
            if ys[i] == 0.8:
                most_probably_x.append(xs[i])
            if ys[i] == 0.2:
                low_probably_x.append(xs[i])
    plt.plot(xs, ys)
    plt.show()
    print(f"Max = {max_x}\nMin = {min_x}")
    print(f"Most probably values: from {most_probably_x[0]} to {most_probably_x[-1]}")
    print(f"Most probably values: from {low_probably_x[0]} to {low_probably_x[-1]}")


if __name__ == '__main__':
    exercise1()
    exercise2()
