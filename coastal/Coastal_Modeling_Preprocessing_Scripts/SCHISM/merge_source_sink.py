#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:15:15 2021

@author: Camaron.George
"""
import os
import argparse
import pathlib
import numpy as np
import netCDF4 as nc
import xarray as xr
import gc

parser = argparse.ArgumentParser()
parser.add_argument('filePath', type=str, help='path to the discharge files')
parser.add_argument('sfluxfile', type=str, help='path to the sflux2sourceInput.nc file')
parser.add_argument('domain', type=str, help='domain name, one of hawaii, prvi, pacfic, or atlgulf')
args = parser.parse_args()

#path to where the discharge files are stored
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Hawaii/SCHISM/POIs_No_Anamolies')
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Alaska/SCHISM/')
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/PR+USVI/SCHISM/')
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Champlain/SCHISM/')
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Pacific/SCHISM/')
filePath = pathlib.Path(args.filePath)
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Erie/SCHISM/')
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Michigan-Huron/SCHISM/')
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Superior/SCHISM/')
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Ontario/SCHISM/')

#path to the sflux2sourceInput.nc file
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/Jason.Ducker/Hawaii_Data/SCHISM/sflux2sourceInput.nc')
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Alaska/SCHISM/sflux2sourceInput.nc')
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/PR+USVI/SCHISM/sflux2sourceInput.nc')
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Champlain/SCHISM/sflux2sourceInput.nc')
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Pacific/SCHISM/sflux2sourceInput.nc')
areaFile = pathlib.Path(args.sfluxfile)
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Erie/SCHISM/sflux2sourceInput.nc')
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Michigan-Huron/SCHISM/sflux2sourceInput.nc')
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Superior/SCHISM/sflux2sourceInput.nc')
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Ontario/SCHISM/sflux2sourceInput.nc')


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
#    next(f)
    nsiel = int(f.readline())
    if(nsiel != 0):
        for i in range(nsiel):
            data = f.readline().split()
            siel.append(int(data[0]))
            troute_sink.append(str(data[1]))

# read in discharge files from combine_sink_source.py
#with open(filePath/'source_sink_BMI.in') as f:
#    nsoel1 = int(f.readline())
#    # Transform source element ids to indexes
#    soel1 = np.loadtxt(f, dtype=str, max_rows=nsoel1) - 1
#    #next(f)
#    nsiel = int(f.readline())
#    # Transform sink element ids to indexes
#    siel = np.loadtxt(f, dtype=str, max_rows=nsiel) - 1

with xr.open_dataset(filePath/'source_TRoute.nc') as TRoute:
    time = TRoute['time_vsource'].data
    si = TRoute['vsink'].data
    so1 = TRoute['vsource'].data

# read in element areas/density and calculate the threshold for each element
#  (threshold results in less than 1 cm change in water level)
# This is used to filter out elements that will not see enough discharge
# to cause flooding. Used to reduce file size.

ncareas = xr.open_dataset(areaFile)
thresholds = (0.02 * ncareas['precip2flux'] * 1000) / time[-1]
thresholds = thresholds.values
#with xr.open_dataset(areaFile) as ncareas:
#    thresholds = (0.01 * ncareas['precip2flux'] * 1000) / time[-1]
#    thresholds = thresholds.values

#find the max discharge for all elements and remove those
#elements where the max discharge is below the threshold
with xr.open_dataset(filePath/'source2.nc') as precip:
    md = precip['vsource'].max(axis=0)
    md[soel1] = (precip['vsource'][:, soel1] + so1).max(axis=0)
    keep = md > thresholds
    so2 = precip['vsource'][:, keep]
    keep_idxs = np.nonzero(keep.values)[0]+1

#md = [so2[:,i].max() for i in range(so2.shape[1])]
#keep = md > threshold
#keep  = [idx for idx,val in enumerate(md) if val > thresholds[idx]]
#so2 = so2[:,keep]
#add one to each index in keep variable to find element numbers of sources
#keep = [keep[i] + 1 for i in range(len(keep))]

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
del keep_idxs
del siel
del md
del thresholds
gc.collect()
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
