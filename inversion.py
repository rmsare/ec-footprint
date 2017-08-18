"""
Least squares inversion for surface flux using footprint functions and measured atmospheric fluxes  
"""

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import lil_matrix, issparse

from cvxpy import *

from footprint import KljunFootprint

BOUNDARY_LAYER_HEIGHT = 1000 # m
EC_STATION_HEIGHT = 4.1 # m
ROUGHNESS_LENGTH_SCALE = 0.03 # m

def calculate_footprint(data, i):
    zm = EC_STATION_HEIGHT
    zo = ROUGHNESS_LENGTH_SCALE
    h = BOUNDARY_LAYER_HEIGHT
    L = data['L'][i]
    sigma_v = data['sigma_v'][i]
    ustar = data['ustar'][i]
    wind_dir = data['wind_dir'][i]
    
    footprint_obj = KljunFootprint(zo, zm, L, sigma_v, ustar, wind_dir, h, 600)
    footprint_obj.calculate_footprint()
    footprint_obj.dx = np.mean(np.diff(footprint_obj.X))
    
    return footprint_obj

def form_G_from_footprints(data, nobs=None, grid_shape=(1001, 1501)):
    if not nobs:
        nobs = len(data['flux'])

    nrows, ncols = grid_shape 
    G = lil_matrix((nobs, nrows*ncols))
    for i in range(0, nobs):
        footprint_obj = calculate_footprint(data, i)
        G[i] = footprint_obj.footprint.reshape((1, nrows*ncols))
        #G[i] = footprint_obj.footprint.reshape((1, nrows*ncols)) / footprint_obj.dx**2

    return G

def invert_data(G, d, dx, thresh=20000):
    n = G.shape[1]

    if issparse(G):
        G = Constant(G)

    m = Variable(n)
    loss = norm(G*m - d, 2)

    lambd = Parameter(sign="positive")
    laplacian = diff(m, k=2) / dx**2
    reg = norm(laplacian, 2)
    
    objective = Minimize(loss + lamdb*red)
    constraints = [m >= 0, 
                   m <= thresh]
    
    problem = Problem(objective, constraints)
    problem.solve()

    return m.value()
