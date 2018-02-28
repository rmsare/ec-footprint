import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from progressbar import Bar, ETA, ProgressBar

from footprint import KljunFootprint as Footprint
from kljun.calc_footprint_FFP_climatology import FFP_climatology


def calculate_footprint(start, stop, directory=''):

    zm = 6. # m 
    h = 1000 # m
    z0 = 0.03 # m 
    nx, ny = (1501, 1001)
    
    data = pd.read_pickle('data/EC_HSL_02252018.pk')

    time_mask = (data.index >= start) & (data.index <= stop) 
    
    valid_mask = (data['wind_dir'] >= 0) & (data['wind_dir'] <= 360) & (data['u*'] >= 0.1) & data['daytime']
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
    
    print('-' * 80)
    print('Calculating footprint climatology for {} to {}...'.format(start, stop))
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
                          verbosity=0)

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
    cbar = plt.colorbar(p, aspect=10, shrink=0.5, orientation='horizontal')
    cbar.set_label('Flux contribution [m$^{-2}$]', fontsize=12)
    
    ax = plt.gca()
    ax.set_title('{} to {}'.format(date0, date1))
    ax.set_xlabel('E [m]', fontsize=14)
    ax.set_ylabel('N [m]', fontsize=14)
    ax.set_aspect(1)

    start = start.strftime('%Y-%m-%d')
    stop = stop.strftime('%Y-%m-%d')
    plt.savefig(directory + 'footprint_{}_{}.png'.format(start, stop), dpi=300)
    np.save(directory + 'footprint_data_{}_{}.npy'.format(start, stop), [X, Y, climatology])

    plt.show()


if __name__ == "__main__":
    years = [2014, 2015, 2016, 2017]
    months = np.linspace(1, 12, 12, dtype=int)

    for year in years:
        start = pd.datetime(year, 1, 1)
        stop = pd.datetime(year + 1, 1, 1)
        calculate_footprint(start, stop, directory='annual/')
    
    for year in years:
        for month in months:
            start = pd.datetime(year, month, 1)
            stop = pd.datetime(year, month, 31)
            calculate_footprint(start, stop, directory='monthly/')
