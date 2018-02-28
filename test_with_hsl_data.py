import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from progressbar import Bar, ETA, ProgressBar

from footprint import KljunFootprint as Footprint
from kljun.calc_footprint_FFP_climatology import FFP_climatology

if __name__ == "__main__":

    zm = 6. # m 
    h = 1000 # m
    z0 = 0.03 # m 
    nx, ny = (1501, 1001)
    
    data = pd.read_pickle('data/EC_HSL_02252018.pk')

    #n_days = 1
    #i1 = int(24 * 2 * n_days) 

    i0 = 0
    i1 = data.shape[0] - 1
    indices = np.linspace(i0, i1, i1 - i0).astype(int)

    date0 = data.index[i0]
    date1 = data.index[i1]

    n = 0

    print('Calculating footprint climatology for {} to {}...'.format(date0, date1))

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
    
    print('Using {}/{} observations'.format(n, len(indices)))

    zms = list(zm * np.ones_like(Ls))
    z0s = list(z0 * np.ones_like(Ls))
    hs = list(h * np.ones_like(Ls))
    
    rs = [0.5, 0.75, 0.9]
    
    res = FFP_climatology(zm=zms, 
                          z0=z0s, 
                          h=hs, 
                          ol=Ls, 
                          sigmav=sigma_vs, 
                          ustar=u_stars, 
                          wind_dir=wind_dirs,
                          rs=rs,
                          crop=True,
                          pulse=1000)

    X = res['x_2d']
    Y = res['y_2d']
    climatology = res['fclim_2d']
    
    Xc = res['xr']
    Yc = res['yr']

    plt.figure()
    p = plt.pcolormesh(X, Y, climatology)
    for x, y, r in zip(Xc, Yc, rs):
        plt.plot(x, y, 'w-')
        plt.text(x[-1], y[-1] + 7.5, '{:.0f}'.format(100 * r), color='w', fontsize=10)
    cbar = plt.colorbar(p, aspect=10, shrink=0.75, orientation='horizontal')
    cbar.set_label('Flux contribution [m$^{-2}$]', fontsize=12)
    
    ax = plt.gca()
    ax.set_title('{} to {}'.format(date0, date1))
    ax.set_xlabel('E [m]', fontsize=12)
    ax.set_ylabel('N [m]', fontsize=12)
    ax.set_aspect(1)

    plt.savefig('footprint_full.png', dpi=300)
    np.save('footprint_data_full.npy', [X, Y, climatology])

    plt.show()
