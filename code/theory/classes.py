from collections import namedtuple
from math import sqrt

from sympy import symbols


class Vector(namedtuple("Vector", "x y z")):
    @staticmethod
    def by_name(name: str, *args, **kwargs):
        sep = '.'
        if not args and not kwargs:
            vector = Vector(*symbols(f'{name}{sep}x,{name}{sep}y,{name}{sep}z'))
        else:
            vector = Vector(*args, **kwargs)
        vector.name = 'name'
        return vector

    def magnitude(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


Interval = namedtuple("Interval", "left right")


class Problem:
    def __init__(self, r, d, h):
        self.r = r
        self.d = d
        self.h = h

    def get_a_by_c(self, c: Vector):
        return self._get_a_by_c(c, self.h, self.d, self.r)

    @staticmethod
    def _get_a_by_c(c: Vector, h, d, r):
        x = (h - 2 * c.z) * c.y + 2 * c.x * c.y * c.z * (r + d) / r ** 2
        x /= (h - c.z)
        y = (h - 2 * c.z) * c.y + (c.y ** 2 - c.x ** 2) * c.z * (r + d) / r ** 2
        y /= (h - c.z)
        return Problem.get_a(x=x, y=y)

    def get_e_by_c(self, c: Vector):
        return self._get_e_by_c(c, self.h, self.d, self.r)

    @staticmethod
    def _get_e_by_c(c: Vector, h, d, r):
        x = (c.x * d) / (r + d - c.y)
        z = ((r - c.y) * h + c.z * d) / (r + d - c.y)
        return Problem.get_e(x=x, z=z, r=r)

    def get_c_x_interval(self):
        r, d = self.r, self.d
        c_y_min = r ** 2 / (r + d)
        c_x_max = sqrt(r ** 2 - c_y_min ** 2)
        return Interval(left=-c_x_max, right=c_x_max)

    def get_c_z_interval(self):
        return Interval(left=0, right=self.h * 0.8)

    def get_c(self, x, z):
        return Vector.by_name(name='C', x=x, y=(self.r ** 2 - x ** 2) ** 0.5, z=z)

    @staticmethod
    def get_a(x, y):
        return Vector.by_name(name='A', x=x, y=y, z=0)

    @staticmethod
    def get_e(x, z, r):
        return Vector.by_name(name='E', x=x, y=r, z=z)



