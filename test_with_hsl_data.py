import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from progressbar import Bar, ETA, ProgressBar

from footprint import KljunFootprint as Footprint
from kljun.calc_footprint_FFP_climatology import FFP_climatology

if __name__ == "__main__":

    zm = 4.1 # m 
    h = 100 # m
    z0 = 0.03 # m 
    nx, ny = (1501, 1001)
    
    data = pd.read_pickle('data/EC_HSL_02252018.pk')

    i0 = 0
    i1 = data.shape[0] - 1
    i1 = 1000
    #i1 = int(24 * 2 * 365 / 64) 
    indices = np.linspace(i0, i1, i1 - i0).astype(int)

    date0 = data.index[i0]
    date1 = data.index[i1]

    n = 0

    print('Calculating footprint climatology for {} to {}...\n'.format(date0, date1))

    Ls = []
    sigma_vs = []
    wind_dirs = []
    u_stars = []

    for i, idx in enumerate(indices):
        wind_dir = data.iloc[idx]['wind_dir']
        u_star = data.iloc[idx]['u*']

        if wind_dir >=0 and wind_dir <= 360 and u_star >= 0.1:
            L = data.iloc[idx]['L']
            v_var = data.iloc[idx]['v_var']
            sigma_v = np.sqrt(v_var)
            
            wind_dirs.append(wind_dir)
            u_stars.append(u_star)
            Ls.append(L)
            sigma_vs.append(sigma_v)
            n += 1
    
    print('Using {} observations'.format(n))

    zms = list(zm * np.ones_like(Ls))
    z0s = list(z0 * np.ones_like(Ls))
    hs = list(h * np.ones_like(Ls))
    
    res = FFP_climatology(zm=zms, 
                          z0=z0s, 
                          h=hs, 
                          ol=Ls, 
                          sigmav=sigma_vs, 
                          ustar=u_stars, 
                          wind_dir=wind_dirs,
                          crop=True,
                          pulse=1000)

    X = res['x_2d']
    Y = res['y_2d']
    climatology = res['fclim_2d']

    plt.figure()
    p = plt.pcolormesh(X, Y, climatology)
    #p = plt.contourf(X, Y, climatology, colors='w')
    c = plt.contour(X, Y, climatology, colors='w')
    plt.clabel(c, fmt='%.5f', inline=True, fontsize=12)
    cbar = plt.colorbar(p, orientation='horizontal')
    cbar.set_label('Flux contribution')
    
    ax = plt.gca()
    ax.set_title('{} to {}'.format(date0, date1))
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect(1)
    
    plt.show()
