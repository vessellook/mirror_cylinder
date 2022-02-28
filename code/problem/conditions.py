from collections import namedtuple

from sympy import symbols, Matrix


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


h, r, d = symbols('h,r,d')
C = Vector.by_name('C')
A = Vector.by_name('A')
E = Vector.by_name('E')
B = Vector.by_name('B', 0, 0, C.z)
F = Vector.by_name('F', 0, r + d, h)

AC = (C - A).magnitude
CF = (C - F).magnitude

equalities = (
    C.x ** 2 + C.y ** 2 - r ** 2,
    Matrix(list(zip(F - B, A - B, C - B))).det(),
    AC / CF - C.z / (h - C.z),
    A.z,
    E.y - r
)
inequalities = (
    r > 0,
    h > 0,
    d > 0,
    A.x ** 2 + A.y ** 2 > 0,
    C.z < h,
    C.z > 0
)
