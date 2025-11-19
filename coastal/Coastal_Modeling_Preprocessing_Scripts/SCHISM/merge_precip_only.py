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

parser = argparse.ArgumentParser()
parser.add_argument('sfluxfile', type=str, help='schism sflux2sourceInput.nc file')
parser.add_argument('source2file', type=str, help='schism source2.nc file')
parser.add_argument('polygon_enclosure', type=str, help='polygon_enclosure file')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('domain', type=str, help='domain name, one of hawaii, prvi, pacfic, or atlgulf')
args = parser.parse_args()

#path to where the discharge files are stored
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Alaska/SCHISM/')
#filePath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/PR+USVI/SCHISM/')
#filePath = pathlib.Path(args.output_dir)

#path to the sflux2sourceInput.nc file
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Alaska/SCHISM/sflux2sourceInput.nc')
#areaFile = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/PR+USVI/SCHISM/sflux2sourceInput.nc')
areaFile = pathlib.Path(args.sfluxfile)
# path for source.nc output
#outPath = pathlib.Path('/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Alaska/SCHISM/')
outPath = pathlib.Path(args.output_dir)

#END USER INPUT#

# Read in the source.nc file quickly and just calculate the time series that
# was originally coming from the source_sink.in file
source = nc.Dataset(args.source2file)
time_length = source.variables['vsource'].shape[0]
source.close()
time = np.arange(0,time_length)*3600.0
# read in element areas/density and calculate the threshold for each element
#  (threshold results in less than 1 cm change in water level)

ncareas = xr.open_dataset(areaFile)
thresholds = (0.01 * ncareas['precip2flux'] * 1000) / time[-1]
thresholds = thresholds.values
#with xr.open_dataset(areaFile) as ncareas:
#    thresholds = (0.01 * ncareas['precip2flux'] * 1000) / time[-1]
#    thresholds = thresholds.values

#find the max discharge for all elements and remove those
#elements where the max discharge is below the threshold
with xr.open_dataset(args.source2file) as precip:
    md = precip['vsource'].max(axis=0)
    keep = md > thresholds
    so2 = precip['vsource'][:, keep]
    keep_idxs = np.nonzero(keep.values)[0]

#md = [so2[:,i].max() for i in range(so2.shape[1])]
#keep = md > threshold
#keep  = [idx for idx,val in enumerate(md) if val > thresholds[idx]]
#so2 = so2[:,keep]
#add one to each index in keep variable to find element numbers of sources
#keep = [keep[i] + 1 for i in range(len(keep))]

mso = np.zeros((time_length,2,len(keep_idxs)))
mso[:,0,:] = -9999

if (outPath/'source_precip_only.nc').exists():
    os.remove(outPath/'source_precip_only.nc')
 
#write source.nc file      
ncout = nc.Dataset(outPath/'source_precip_only.nc','w',format='NETCDF4')

ncout.createDimension('time_vsource',len(time))
ncout.createDimension('time_vsink',len(time))
ncout.createDimension('time_msource',len(time))
ncout.createDimension('nsources',len(keep_idxs))
ncout.createDimension('nsinks',0)
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
ncvso[:] = so2
ncvmo[:] = mso
nctso[:] = time
nctsi[:] = time
nctmo[:] = time
ncvsos[:] = time[1] - time[0]
ncvsis[:] = time[1] - time[0]
ncvmos[:] = time[1] - time[0]

ncout.close()
