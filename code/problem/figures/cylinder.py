"""
This module contains class PhysicalRay that represents a ray passed from point of view F 
to point of mirror cylinder C and point of xOy surface A. Also it contains point E
"""
import numpy as np
import plotly.graph_objects as go

from problem.vector import Vector


class Cylinder:
    def __init__(self, *, r, h, d=None):
        self.r = r
        self.h = h
        self.d = d

    def points(self, n=100, m=100):
        return map(Vector._make, zip(*self.coord_lists(n, m)))

    def coord_lists(self, n=100, m=100):
        if self.d is None:
            theta = np.linspace(0, 2 * np.pi, n)
        else:
            min_cos = self.r / (self.r + self.d)
            theta_border = np.arccos(min_cos)
            theta = np.linspace(-theta_border, theta_border, n)
        v = np.linspace(0, self.h, m)
        x = self.r * np.outer(np.sin(theta), np.ones(m))
        y = self.r * np.outer(np.cos(theta), np.ones(m))
        z = np.outer(np.ones(n), v)
        return x, y, z

    def surface(self, n=100, m=100):
        colorscale = [[0, 'blue'], [1, 'blue']]
        x, y, z = self.coord_lists(n, m)
        return go.Surface(x=x, y=y, z=z,
                          colorscale=colorscale,
                          showscale=False,
                          opacity=0.8,
                          customdata=['C'] * len(x))
