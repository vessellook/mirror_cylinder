from math import log
from pprint import pprint
from itertools import product

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

from theory.classes import Vector, Problem
from theory.functions import get_c_x_range, get_c_z_range


def func(X, y):
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    return reg


class get_c_by_a:
    def __init__(self, a_values, c_values):
        X = np.array(list(map(g, a_values)))
        self.x = func(X, np.array([c.x for c in c_values]))
        self.y = func(X, np.array([c.y for c in c_values]))
        self.z = func(X, np.array([c.z for c in c_values]))

    def __call__(self, a: Vector):
        x = self.x.predict([self.get_X(a)])[0]
        y = self.y.predict([self.get_X(a)])[0]
        z = self.z.predict([self.get_X(a)])[0]
        return Vector.by_name(name='C', x=x, y=y, z=z)

    @staticmethod
    def get_X(a: Vector):
        return 1, a.x, a.y, a.x * a.y, a.x / a.y, a.x ** 2, a.y ** 2, 1 / (a.x + 0.0001)


# def get_c_by_a(a_values, c_values):
#     g = lambda a: (a.x, a.y, a.x * a.y, a.x / a.y, a.x ** 2, a.y ** 2, 1 / (a.x + 0.0001), 1)
#     X = np.array(list(map(g, a_values)))
#     reg_x = func(X, np.array([c.x for c in c_values]))
#     reg_y = func(X, np.array([c.y for c in c_values]))
#     reg_z = func(X, np.array([c.z for c in c_values]))
#
#     def reg(a: Vector):
#         x = reg_x.predict([g(a)])[0]
#         y = reg_y.predict([g(a)])[0]
#         z = reg_z.predict([g(a)])[0]
#         return Vector.by_name(name='C', x=x, y=y, z=z)
#
#     reg.x = reg_x
#     reg.y = reg_y
#     reg.z = reg_z
#
#     return reg


def generate_vectors(p: Problem, n=100, m=100):
    c_values = [p.get_c(x=x, z=z) for x, z in product(get_c_x_range(n, p.d, p.r), get_c_z_range(m, p.h))]
    a_values = list(map(p.get_a_by_c, c_values))
    e_values = list(map(p.get_e_by_c, c_values))

    # for c, a, e in zip(c_values, a_values, e_values):
    #     print('C =', c)
    #     print('A =', a)
    #     print('E =', e)
    #     print()

    return c_values, a_values, e_values


def generate_vectors2(p: Problem, n=10, m=10):
    c_values = [p.get_c(x=x, z=z) for x, z in product(get_c_x_range(n, p.d, p.r), get_c_z_range(m, p.h))]
    a_values = list(map(p.get_a_by_c, c_values))
    e_values = list(map(p.get_e_by_c, c_values))

    # for c, a, e in zip(c_values, a_values, e_values):
    #     print('C =', c)
    #     print('A =', a)
    #     print('E =', e)
    #     print()

    return c_values, a_values, e_values


def main():
    h = 2
    d = 1
    r = 1

    p = Problem(r=r, d=d, h=h)
    c_values, a_values, e_values = generate_vectors(p)

    f = get_c_by_a(a_values, c_values)
    false_c_values = list(map(f, a_values))
    a = sum((c_.x - c.x) ** 2 for c_, c in zip(false_c_values, c_values)) / len(c_values)
    b = sum((c_.y - c.y) ** 2 for c_, c in zip(false_c_values, c_values)) / len(c_values)
    k = sum((c_.z - c.z) ** 2 for c_, c in zip(false_c_values, c_values)) / len(c_values)
    pprint([a, b, k])

    # c_values, a_values, e_values = generate_vectors2(p, n=1000)
    # points = list(map(lambda v: v.z, a_values))
    # plt.plot(list(map(lambda v: v.x, a_values)), list(map(lambda v: v.x, c_values)), 'r.')
    # plt.plot(list(map(lambda v: v.y, a_values)), list(map(lambda v: v.x, c_values)), 'g.')
    # plt.plot(list(map(lambda v: v.y, a_values)), list(map(lambda v: v.x, c_values)), 'b.')
    # plt.xlabel('A')
    # plt.ylabel('C.x')
    # plt.show()


if __name__ == '__main__':
    main()
