"""
Footprint function inversion from eddy covariance data
"""

import sys

import numpy as np
import scipy
from scipy.special import gamma
from scipy.integrate import quad, odeint
from scipy.interpolate import griddata

import matplotlib.pyplot as plt

SHAPE_FACTOR = 1.5
VON_KARMAN_CONSTANT = 0.41


class Footprint(object):

    def __init__(self, x, y, zo, zm, L, ustar, sigma_v, wind_dir):
        self.X, self.Y = np.meshgrid(x, y) 
        self.zo = zo
        self.zm = zm
        self.L = L
        self.ustar = ustar
        self.sigma_v = sigma_v
        self.wind_dir = wind_dir


class HorstWeilFootprint(Footprint):
    
    def __init__(self, x, y, zo, zm, L, ustar, sigma_v, wind_dir):
        #self.X, self.Y = np.meshgrid(x, y) 
        self.X = x
        self.Y = y
        self.zo = zo
        self.zm = zm
        self.L = L
        self.ustar = ustar
        self.sigma_v = sigma_v
        self.wind_dir = wind_dir

        self.r = SHAPE_FACTOR
        self.k = VON_KARMAN_CONSTANT

        self.A = self.r * gamma(2 / self.r) / gamma(1 / self.r)**2
        self.b = gamma(2 / self.r) / gamma(1 / self.r)
        self.p = ((self.r * gamma(2 / self.r) / gamma(1 / self.r))**self.r)**(1 / (1 - self.r))

    def Dy(self, x):
        n = len(x)
        Y2 = (self.Y**2).reshape((n, 1))
        return (1 / (np.sqrt(2 * np.pi) * self.sigma_y(x))) * np.exp(-Y2 / (2 * self.sigma_y(x)**2))
    
    def dzbar_dx(self, x, zbar):
        return self.k**2 / ((np.log(self.p * zbar / self.zo) - self.psi(self.p * zbar, self.L)) * self.phic(self.p * zbar, self.L))

    def calculate_footprint(self):
        self.footprint = self.Dy(self.X) * self.fybar(self.X)

    def sigma_y(self, x):
        sigma_y = self.sigma_v * x / self.U(x) 
        return sigma_y

    def fybar(self, x):
        """
        Horst and Weil 1991
        Equation 15
        """
        return (self.A / self.zbar(x)) * self.dzbar_dx(x, self.zbar(x)) * (self.zm / self.zbar(x)) * np.exp((-self.zm / (self.b * self.zbar(x)))**self.r)

    def set_flux_distribution(self, flux):
        pass
    
    def phic(self, z, L):
        """
        Horst and Weil 1991
        Appendix
        """

        if 1 / L >= 0:
            phic = 1 + 5 * z / L
        else:
            phic = (1 - 16 * z / L)**-0.5

        return phic

    def psi(self, z, L):
        """
        Horst and Weil 1991
        Appendix
        """

        if 1 / L >= 0:
            psi = -5 * z / L
        else:
            x = (1 - 16 * z / L)**0.25
            psi = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan2(x, 1) + np.pi / 2

        return psi
        
    def rotate_footprint(self, theta=None):
        if not theta:
            theta = self.wind_dir

        XY = np.hstack([self.X.ravel(), self.Y.ravel()])
        Xrot = self.X * np.cos(theta) + self.Y * np.sin(theta)
        Yrot = -self.X * np.sin(theta) + self.Y * np.cos(theta)
        XYrot = np.hstack([Xrot.ravel(), Yrot.ravel()])

        return griddata(XY, self.footprint, XYrot)

    def U(self, x):
        dg_dz = lambda z, x : (self.ustar / self.k) * (np.log(z / self.zo) - self.psi(z, self.L)) * (self.A / self.zbar(x))*np.exp(-(z / (self.b * self.zbar(x)))**self.r)
        return [quad(dg_dz, self.zo, np.inf, args=([xi],))[0] for xi in x] 

    def zbar(self, x):
        if x[0] != 0:
            x = np.insert(x, 0, 0)
        return odeint(self.dzbar_dx, self.zo, x)[1:]


class KljunFootprint(Footprint):

    def __init__(self, zo, zm, L, ustar, sigma_v, wind_dir, h, nx=1000):
        #self.nx = len(x)
        self.nx = nx 
        self.zo = zo
        self.zm = zm
        self.L = L
        self.ustar = ustar
        self.sigma_v = sigma_v
        self.wind_dir = wind_dir
        self.h = h

    def calculate_footprint(self):
        # Uses Kljun et al 2015 footprint estimate
        result = FFP(zm=self.zm, z0=self.zo, h=self.h, ol=self.L, sigmav=self.sigma_v, ustar=self.ustar, wind_dir=self.wind_dir, nx=self.nx)
        self.X = result['x_2d']
        self.Y = result['y_2d']
        self.footprint = result['f_2d']
        self.rotate_footprint()

    def rotate_footprint(self):
        """
        Rotates footprint returned by Kljun et al 2015, which supplies rotated
        coordinate grids with a footprint referenced to wind direction = 0
        
        """
        ny, nx = self.footprint.shape
        x = np.linspace(-nx/2, nx/2, nx)
        y = np.linspace(-ny/2, ny/2, ny)
        Xorig, Yorig = np.meshgrid(x, y) 
        s = np.prod(Xorig.shape)
        XY = np.hstack([Xorig.reshape((s,1)), Yorig.reshape((s,1))])
        footprint = self.footprint.reshape((s,1))

        s = np.prod(self.X.shape)
        XYrot = np.hstack([self.X.reshape((s,1)), self.Y.reshape((s,1))])

        self.footprint = griddata(XYrot, footprint, XY).reshape((ny, nx))
        self.footprint[np.isnan(self.footprint)] = 0

class KormannMeixnerFootprint(Footprint):
    pass


