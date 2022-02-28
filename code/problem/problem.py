from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from problem.constants import COORD_NAMES, A_COORD_NAMES, C_COORD_NAMES, E_COORD_NAMES, ALMOST_ZERO
from problem.vector import vector_f, Vector, vector_c, vector_a, vector_e
from problem.vector_function import (VectorFunction, VectorFunctionC2E, VectorFunctionC2A, CompositeVectorFunction,
                                     VectorRegressor)
from problem.figures.ray import Ray


def _convert_vector_data(v):
    x, y, z = v.x, v.y, v.z
    return (
        x * 0 + 1,
        x,
        y,
        z,
        np.square(x),
        np.square(y),
        np.square(z),
        1 / (x + ALMOST_ZERO),
        1 / (y + ALMOST_ZERO),
        1 / (z + ALMOST_ZERO),
        x * y,
        x * z,
        y * z,
        np.arctan(y / x),
    )


@dataclass
class Problem:
    r: Real
    d: Real
    h: Real
    f : Vector = field(init=False)
    c2a : VectorFunction = field(init=False)
    c2e : VectorFunction = field(init=False)

    def __post_init__(self):
        self.f = vector_f(r=self.r, d=self.d, h=self.h)
        self.c2a = VectorFunctionC2A(r=self.r, d=self.d, h=self.h,
                                     src_coord_names=dict(
                                         zip(COORD_NAMES, C_COORD_NAMES)),
                                     dest_coord_names=dict(zip(COORD_NAMES, A_COORD_NAMES)))
        self.c2e = VectorFunctionC2E(r=self.r, d=self.d, h=self.h,
                                     src_coord_names=dict(
                                         zip(COORD_NAMES, C_COORD_NAMES)),
                                     dest_coord_names=dict(zip(COORD_NAMES, E_COORD_NAMES)))

    def get_ray(self, *, a: Vector = None, c: Vector = None, e: Vector = None):
        if isinstance(c, Vector):
            return Ray(a=self.c2a(c), c=c, e=self.c2e(c), f=self.f)
        elif isinstance(a, Vector):
            return Ray(a=a, c=self.a2c(a), e=self.a2e(a), f=self.f)
        elif isinstance(e, Vector):
            return Ray(a=self.e2a(e), c=self.e2c(e), e=e, f=self.f)
        else:
            raise AttributeError(
                'At least one vector from a, c, e should be passed as param')

    @cached_property
    def a2c(self):
        src_coord_names = dict(zip(COORD_NAMES, A_COORD_NAMES))
        dest_coord_names = dict(zip(COORD_NAMES, C_COORD_NAMES))
        func = VectorRegressor([DecisionTreeRegressor() for _ in range(3)],
                               convert_data=_convert_vector_data,
                               src_coord_names=src_coord_names,
                               dest_coord_names=dest_coord_names)
        df = self.get_dataframe()
        c = df.rename(columns={v: k for k, v in dest_coord_names.items()})
        a = df.rename(columns={v: k for k, v in src_coord_names.items()})

        func.fit(X=a, Y=c)
        return func

    @cached_property
    def e2c(self):
        src_coord_names = dict(zip(COORD_NAMES, E_COORD_NAMES))
        dest_coord_names = dict(zip(COORD_NAMES, C_COORD_NAMES))
        func = VectorRegressor([DecisionTreeRegressor() for _ in range(3)],
                               convert_data=_convert_vector_data,
                               src_coord_names=src_coord_names,
                               dest_coord_names=dest_coord_names)
        df = self.get_dataframe()
        c = df.rename(columns={v: k for k, v in dest_coord_names.items()})
        e = df.rename(columns={v: k for k, v in src_coord_names.items()})

        func.fit(X=e, Y=c)
        return func

    @cached_property
    def a2e(self):
        return CompositeVectorFunction(vector_functions=[self.a2c, self.c2e])

    @cached_property
    def e2a(self):
        return CompositeVectorFunction(vector_functions=[self.e2c, self.c2a])

    def get_c(self, *, x=None, y=None, z):
        return vector_c(x=x, y=y, z=z, r=self.r)

    @staticmethod
    def get_a(x, y):
        return vector_a(x=x, y=y)

    def get_e(self, x, z):
        return vector_e(x=x, z=z, r=self.r)

    def get_c_dataframe(self, n, m):
        r, d, h = self.r, self.d, self.h

        cos_min = r / (r + d)
        theta_min = np.arccos(cos_min)
        theta = np.linspace(-theta_min, theta_min, n)
        cz = np.linspace(0, h*0.8, m)
        theta, cz = np.meshgrid(theta, cz)
        cx = r * np.sin(theta)
        cy = r * np.cos(theta)

        for array in (cx, cy, cz):
            array.shape = n * m
        return pd.DataFrame(dict(zip(C_COORD_NAMES, (cx, cy, cz))))

    def get_dataframe(self, n=100, m=100):
        c = self.get_c_dataframe(n, m)
        a = self.c2a(c)
        e = self.c2e(c)

        return pd.concat([a, c, e], axis=1)
