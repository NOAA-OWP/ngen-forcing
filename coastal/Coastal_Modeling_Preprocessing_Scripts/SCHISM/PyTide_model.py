import os
import netCDF4
import numpy as np
from scipy.interpolate import griddata

def PyTide_model(consts_path, schism_x_bnd, schism_y_bnd):
    tide_constants = ['k1', 'k2', 'm2', 'n2', 'o1', 'p1', 'q1', 's2']
    consfiles = list(map(lambda x: os.path.join(consts_path, x)+".nc", tide_constants))
    pha = np.zeros((len(schism_x_bnd), len(consfiles)))
    amp = np.zeros((len(schism_x_bnd), len(consfiles)))

    col = 0
    for file in consfiles:
        data = netCDF4.Dataset(file, 'r')
        if col == 0:
            x = data.variables['lon'][:]
            y = data.variables['lat'][:]
            x, y = np.meshgrid(x, y)
            i = np.where(x > 180.0)
            x[i] = x[i]-360.0
            i = np.where((x < max(schism_x_bnd)+1) & (x > min(schism_x_bnd)-1) & (y < max(schism_y_bnd)+1) & (y > min(schism_y_bnd)-1))
            x = x[i]
            y = y[i]

        a = data.variables['amplitude'][:]
        a = a[i]
        p = data.variables['phase'][:]
        p = p[i]

        xI = x[a.mask == False]
        yI = y[a.mask == False]
        p = p[a.mask == False]
        a = a[a.mask == False]

        amp[:, col] = griddata((xI, yI), a, (schism_x_bnd, schism_y_bnd), method='nearest')
        pha[:, col] = griddata((xI, yI), p, (schism_x_bnd, schism_y_bnd), method='nearest')
        col += 1

    return amp, pha

