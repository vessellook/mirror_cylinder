"""
This module contains class PhysicalRay that represents a ray passed from point of view F 
to point of mirror cylinder C and point of xOy surface A. Also it contains point E
"""

from dataclasses import dataclass

import numpy as np
from plotly.graph_objects import Scatter3d

from problem.vector import Vector


@dataclass
class Ray:
    a: Vector
    c: Vector
    e: Vector
    f: Vector
    
    def trace(self, n: int, m: int = None):
        if m is None:
            m = n
        a, c, f = self.a, self.c, self.f
        return Scatter3d(x=np.concatenate((np.linspace(f.x, c.x, n), np.linspace(c.x, a.x, m))),
                         y=np.concatenate((np.linspace(f.y, c.y, n), np.linspace(c.y, a.y, m))),
                         z=np.concatenate((np.linspace(f.z, c.z, n), np.linspace(c.z, a.z, m))),
                         line=dict(color='yellow', width=2), mode='lines')
