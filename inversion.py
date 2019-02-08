"""
Least squares inversion for surface flux using footprint functions and measured atmospheric fluxes and covariates  
"""

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import lil_matrix, issparse

from cvxpy import diff, norm
from cvxpy import Constant, Minimize, Parameter, Problem, Variable  

from footprint import KljunFootprint

BOUNDARY_LAYER_HEIGHT = 2000 # m
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
    
    objective = Minimize(loss + lambd*reg)
    constraints = [m >= 0, 
                   m <= thresh]
    
    problem = Problem(objective, constraints)
    problem.solve()

    return m.value()

def test_climatology(data, nsteps, grid_size):
    climatology = np.zeros(grid_size)
    
    for i in range(nsteps):
        footprint = calculate_footprint(data, i)
        climatology += footprint.footprint

    climatology /= nsteps

    return footprint.X, footprint.Y, climatology


def test_inversion(data):
    
    nobs = len(data['flux'])

    print('Number of observations:\t{}'.format(nobs))

    f = calculate_footprint(data, 0).footprint
    s = f.shape
    
    print('Number of grid points:\t{:d}'.format(np.prod(s)))

    print("Forming G...")
    #G = form_G_from_footprints(data, np.prod(s), grid_shape=s)

    print("Performing inversion...")
    #m = invert_data(G, d, dx)

if __name__ == "__main__":
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer

    from kljun.calc_footprint_FFP_climatology import FFP_climatology

    data = loadmat('HSL2013.mat')
    data['wind_dir'] = data['wd']
    data['sigma_v'] = data['sigv']
    
    nobs = 10 
    
    start = timer()
    #X, Y, climatology = test_climatology(data, nsteps, s)
    zo = list(0.01*np.ones((nobs, 1)))
    zm = list(4.1*np.ones((nobs, 1)))
    h = list(BOUNDARY_LAYER_HEIGHT*np.ones((nobs, 1)))

    results = FFP_climatology(zm, zo, None, h, list(data['L'][:nobs,0]), list(data['sigma_v'][:nobs,0]), list(data['ustar'][:nobs,0]), list(data['wind_dir'][:nobs,0]))
    X = results['x_2d']
    Y = results['y_2d']
    climatology = results['fclim_2d']
    stop = timer()
    print('Runtime: {:.2f} s'.format(stop - start))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.contourf(X, Y, climatology)
    plt.colorbar(im)
