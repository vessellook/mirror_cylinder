import abc
from typing import Callable
from collections.abc import Iterable
from functools import singledispatchmethod

import numpy as np
import pandas as pd

from problem.constants import DEFAULT_COORD_MAP, COORD_NAMES
from problem.vector import Vector, vector_a, vector_e


def c2a(cx, cy, cz, *, h, d, r):
    ax = (h - 2 * cz) * cy + 2 * cx * cy * cz * (r + d) / r ** 2
    ax /= (h - cz)
    ay = (h - 2 * cz) * cy + (cy ** 2 - cx ** 2) * cz * (r + d) / r ** 2
    ay /= (h - cz)
    az = cx * 0
    return ax, ay, az


def c2e(cx, cy, cz, *, h, d, r):
    ex = (cx * d) / (r + d - cy)
    ey = cx * 0
    ez = ((r - cy) * h + cz * d) / (r + d - cy)
    return ex, ey, ez


class VectorFunction(abc.ABC):
    def __init__(self, src_coord_names: dict = DEFAULT_COORD_MAP,
                 dest_coord_names: dict = DEFAULT_COORD_MAP):
        self._src_convert_names = {v: k for k,v in src_coord_names.items()}
        self._dest_coord_names = dest_coord_names
    
    @singledispatchmethod
    def __call__(self, v):
        raise NotImplementedError()

    @__call__.register
    def _(self, v: Vector) -> Vector:
        return self._call_vector(v)
    
    @__call__.register
    def _(self, v: pd.DataFrame) -> pd.DataFrame:
        src = v.rename(columns=self._src_convert_names)
        dest = self._call_dataframe(src)
        return dest.rename(columns=self._dest_coord_names)
    
    @abc.abstractmethod
    def _call_vector(self, v: Vector) -> Vector:
        pass

    @abc.abstractmethod
    def _call_dataframe(self, v: pd.DataFrame) -> pd.DataFrame:
        pass


class CompositeVectorFunction(VectorFunction):
    def __init__(self, vector_functions: Iterable,
                 src_coord_names: dict = DEFAULT_COORD_MAP,
                 dest_coord_names: dict = DEFAULT_COORD_MAP):
        if not vector_functions:
            raise AttributeError(
                'should be at least one item in vector_functions param')
        super().__init__(src_coord_names, dest_coord_names)
        self.vector_functions = vector_functions

    def _call_vector(self, v: Vector) -> Vector:
        for f in self.vector_functions:
            v = f(v)
        return v

    def _call_dataframe(self, v: pd.DataFrame) -> pd.DataFrame:
        for f in self.vector_functions:
            v = f(v)
        return v


class VectorFunctionC2A(VectorFunction):
    def __init__(self, src_coord_names: dict = DEFAULT_COORD_MAP,
                 dest_coord_names: dict = DEFAULT_COORD_MAP, *, h, d, r):
        super().__init__(src_coord_names, dest_coord_names)
        self.h = h
        self.d = d
        self.r = r

    def _call_vector(self, v: Vector) -> Vector:
        return Vector._make(c2a(*v, h=self.h, d=self.d, r=self.r))

    def _call_dataframe(self, v: pd.DataFrame) -> pd.DataFrame:
        ax, ay, az = c2a(cx=v.x, cy=v.y, cz=v.z, h=self.h, r=self.r, d=self.d)
        return pd.concat([ax, ay, az], axis=1, columns=lambda index: COORD_NAMES[index])


class VectorFunctionC2E(VectorFunction):
    def __init__(self, src_coord_names: dict = DEFAULT_COORD_MAP,
                 dest_coord_names: dict = DEFAULT_COORD_MAP, *, h, d, r):
        super().__init__(src_coord_names, dest_coord_names)
        self.h = h
        self.d = d
        self.r = r

    def _call_vector(self, v: Vector) -> Vector:
        return Vector._make(c2e(*v, h=self.h, d=self.d, r=self.r))

    def _call_dataframe(self, v: pd.DataFrame) -> pd.DataFrame:
        ex, ey, ez = c2e(cx=v.x, cy=v.y, cz=v.z, h=self.h, r=self.r, d=self.d)
        return pd.concat([ex, ey, ez], axis=1, columns=lambda index: COORD_NAMES[index])


class VectorRegressor(VectorFunction):
    def __init__(self, regressors: Iterable, convert_data: Callable[[Vector], Iterable],
                 src_coord_names: dict = DEFAULT_COORD_MAP, 
                 dest_coord_names: dict = DEFAULT_COORD_MAP):
        super().__init__(src_coord_names, dest_coord_names)
        self._regressors = regressors
        self._convert_data = convert_data

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        X = pd.DataFrame(self._convert_data(X)).transpose()
        
        for coord, regressor in zip(COORD_NAMES, self._regressors):
            regressor.fit(X, Y[coord])

    def predict(self, X):
        X = pd.DataFrame(self._convert_data(X)).transpose()
        predictions = np.array([regressor.predict(X) for regressor in self._regressors])
        return pd.DataFrame(predictions, index=COORD_NAMES).transpose()

    def _call_vector(self, v: Vector) -> Vector:
        v = self._convert_data(v)
        return Vector._make(regressor.predict([v])[0] for regressor in self._regressors)

    def _call_dataframe(self, v: pd.DataFrame) -> pd.DataFrame:
        return self.predict(v)
