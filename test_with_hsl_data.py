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

    year = 2014
    date0 = pd.datetime(year, 1, 1)
    date1 = pd.datetime(year, 12, 31)
    time_mask = (data.index >= date0) & (data.index <= date1) 
    
    valid_mask = (data['wind_dir'] >= 0) & (data['wind_dir'] <= 360) & (data['u*'] >= 0.1)
    mask = time_mask & valid_mask

    Ls = list(data.loc[mask]['L'])
    sigma_vs = list(np.sqrt(data.loc[mask]['v_var']))
    wind_dirs = list(data.loc[mask]['wind_dir'])
    u_stars = list(data.loc[mask]['u*'])
    zms = list(zm * np.ones_like(Ls))
    z0s = list(z0 * np.ones_like(Ls))
    hs = list(h * np.ones_like(Ls))

    n = len(Ls)
    
    rs = [0.5, 0.75, 0.9]
    
    print('Calculating footprint climatology for {} to {}...'.format(date0, date1))
    print('Using {}/{} observations'.format(n, np.sum(time_mask)))

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

    plt.savefig('footprint_{}.png'.format(year), dpi=300)
    np.save('footprint_data_{}.npy'.format(year), [X, Y, climatology])

    plt.show()
