from collections import namedtuple

from sympy import symbols


class Vector(namedtuple("Vector", "x y z")):
    @staticmethod
    def by_name(name: str, *args, **kwargs):
        if not args and not kwargs:
            sep = '.'
            vector = Vector(*symbols(f'{name}{sep}x,'
                                     f'{name}{sep}y,'
                                     f'{name}{sep}z'))
        else:
            vector = Vector(*args, **kwargs)
        vector.name = 'name'
        return vector

    @property
    def magnitude(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def __add__(self, other):
        return Vector(self.x + other.x,
                      self.y + other.y,
                      self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x,
                      self.y - other.y,
                      self.z - other.z)

def vector_a(x, y):
    return Vector.by_name('A', x, y, 0)


def vector_c(*, x=None, y=None, z, r):
    if x is not None:
        y = (r ** 2 - x ** 2) ** 0.5
    elif y is not None:
        x = (r ** 2 - y ** 2) ** 0.5
    else:
        raise AttributeError('x or y must be not None')
    return Vector.by_name('C', x, y, z)


def vector_e(x, z, *, r):
    return Vector.by_name('E', x, r, z)


def vector_f(*, r, d, h):
    return Vector.by_name('F', 0, r + d, h)


def vector_b(z):
    return Vector.by_name('B', 0, 0, z)
