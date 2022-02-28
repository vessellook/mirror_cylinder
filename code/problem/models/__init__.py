from dataclasses import dataclass
from pathlib import Path
from typing import Union
import pickle
import os

from problem.vector_function import VectorRegressor

__all__ = ['ModelManager']


_dir_path = Path(__file__).parent


@dataclass(frozen=True)
class ModelSettings:
    type: str
    r: float
    h: float
    d: float
    n: int = 100
    m: int = 100

    @property
    def filename(self):
        return f'{self.type}__r={self.r}__h={self.h}__d={self.d}__n={self.n}__m={self.m}.pickle'


class ModelManager:
    def __init__(self, dir_path: Union[Path, str] = None):
        if dir_path is None:
            dir_path = _dir_path
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.dir_path = dir_path

    def __contains__(self, settings: ModelSettings):
        return (self.dir_path / settings.filename).is_file()

    def __getitem__(self, settings: ModelSettings) -> VectorRegressor:
        try:
            with (self.dir_path / settings.filename).open(mode='rb') as file:
                return pickle.load(file)
        except FileNotFoundError as exc:
            raise KeyError from exc

    def __setitem__(self, settings: ModelSettings, model: VectorRegressor):
        with (self.dir_path / settings.filename).open(mode='wb') as file:
            pickle.dump(model, file)

    def __delitem__(self, settings: ModelSettings):
        os.remove(self.dir_path / settings.filename)

    def clear(self):
        for file in self.dir_path.iterdir():
            os.remove(file)