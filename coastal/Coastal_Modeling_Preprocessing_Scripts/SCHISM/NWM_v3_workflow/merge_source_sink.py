#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:15:15 2021

@author: Camaron.George
"""
import os
import pathlib
import numpy as np
import netCDF4 as nc
import xarray as xr

#path to where the discharge files are stored
filePath = pathlib.Path('./NWMv3_Atlantic/')

#path to the sflux2sourceInput.nc file
areaFile = pathlib.Path('./NWMv3_Atlantic/sflux2sourceInput.nc')

soel1 = []
siel = []
troute_source = []
troute_sink = []
with open(os.path.join(filePath,'source_sink_BMI.in')) as f:
    nsoel1 = int(f.readline())
    for i in range(nsoel1):
        data = f.readline().split()
        soel1.append(int(data[0]))
        troute_source.append(str(data[1]))
    next(f)
    nsiel = int(f.readline())
    if(nsiel != 0):
        for i in range(nsiel):
            data = f.readline().split()
            siel.append(int(data[0]))
            troute_sink.append(str(data[1]))


with xr.open_dataset(filePath/'source_NWMv3.nc') as NWM:
    time = NWM['time_vsource'].data
    si = NWM['vsink'].data
    so1 = NWM['vsource'].data

# read in element areas/density and calculate the threshold for each element
#  (threshold results in less than 1 cm change in water level)
# This is used to filter out elements that will not see enough discharge
# to cause flooding. Used to reduce file size.
ncareas = xr.open_dataset(areaFile)
thresholds = (0.01 * ncareas['precip2flux'] * 1000) / time[-1]
thresholds = thresholds.values
ncareas.close()

#find the max discharge for all elements and remove those
#elements where the max discharge is below the threshold
with xr.open_dataset(filePath/'source2.nc') as precip:
    md = precip['vsource'].max(axis=0)
    md[soel1] = (precip['vsource'][:, soel1] + so1).max(axis=0)
    keep = md > thresholds
    so2 = precip['vsource'][:, keep]
    keep_idxs = np.nonzero(keep.values)[0]


mso = np.zeros((len(si),2,len(keep_idxs)))
mso[:,0,:] = -9999

if (filePath/'source.nc').exists():
    os.remove(filePath/'source.nc')
 
#write source.nc file      
ncout = nc.Dataset(filePath/'source.nc','w',format='NETCDF4')

ncout.createDimension('time_vsource',len(time))
ncout.createDimension('time_vsink',len(time))
ncout.createDimension('time_msource',len(time))
ncout.createDimension('nsources',len(keep_idxs))
ncout.createDimension('nsinks',nsiel)
ncout.createDimension('ntracers',2)
ncout.createDimension('one',1)

ncso = ncout.createVariable('source_elem','i4',('nsources',))
ncsi = ncout.createVariable('sink_elem','i4',('nsinks',))
ncvso = ncout.createVariable('vsource','f8',('time_vsource','nsources',))
ncvsi = ncout.createVariable('vsink','f8',('time_vsink','nsinks',))
ncvmo = ncout.createVariable('msource','f8',('time_msource','ntracers','nsources',))
nctso = ncout.createVariable('time_vsource','f8',('time_vsource',))
nctsi = ncout.createVariable('time_vsink','f8',('time_vsink',))
nctmo = ncout.createVariable('time_msource','f8',('time_msource',))
ncvsos = ncout.createVariable('time_step_vsource','f4',('one',))
ncvsis = ncout.createVariable('time_step_vsink','f4',('one',))
ncvmos = ncout.createVariable('time_step_msource','f4',('one',))

ncso[:] = keep_idxs
ncsi[:] = siel
ncvso[:] = so2
ncvsi[:] = si
ncvmo[:] = mso
nctso[:] = time
nctsi[:] = time
nctmo[:] = time
ncvsos[:] = time[1] - time[0]
ncvsis[:] = time[1] - time[0]
ncvmos[:] = time[1] - time[0]

ncout.close()
