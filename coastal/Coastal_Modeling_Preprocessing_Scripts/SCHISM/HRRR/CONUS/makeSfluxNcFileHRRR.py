#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 01:26:35 2022

@author: Camaron.George
"""

import os 
import argparse
import numpy as np 
import netCDF4 as nc
import xarray as xr
import pandas as pd
import geopandas as gpd
from operator import truediv
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
import subprocess
from common.io import read_pli, read_polygon


parser = argparse.ArgumentParser()
parser.add_argument('cppfile', type=str, help='schism hgrid.cpp file')
parser.add_argument('gr3file', type=str, help='schism hgrid.gr3 file')
parser.add_argument('polygon_enclosure', type=str, help='polygon_enclosure file')
parser.add_argument('hrrrfile', type=str, help='a hrrr file')
parser.add_argument('output_dir', type=str, help='output directory for the sflux2sourceInput.nc file')
parser.add_argument('domain', type=str, help='domain name, one of hawaii, prvi, pacfic, or atlgulf')
args = parser.parse_args()

#this script requires hgrid.utm, hgrid.ll, and netcdf precipitation file
#this path should include all of those files; if it doesn't, paths will need to be adjusted below
#path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Pacific_Data/SCHISM'
#path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Atlantic_Data/SCHISM'
#path = '/efs/ngwpc-coastal/parm/coastal/atlgulf'
path = args.output_dir
#cppFile = os.path.join(path,'hgrid.cpp')
#llFile = os.path.join(path,'hgrid.gr3')
outfile = os.path.join(path,'sflux2sourceInput.nc')
cppFile = args.cppfile
llFile=args.gr3file
#polygon_enclosure = '/scratch2/NCEPDEV/ohd/Jason.Ducker/extract4dflow-master/Extract_DFlowFM_Input_Files/Polygon_enclosure_files/SCHISM_Pac_Enclosure.pol'
#polygon_enclosure = '../Coastal_Modeling_Preprocessing_Scripts/Coastal_Data/Enclosure_Polygons/SCHISM_Atl_Enclosure.pol'
polygon_enclosure = args.polygon_enclosure

# Specify where a Alaska HRRR raw grib2 file is on your system
# and it doesn't matter which file it is, just needs to be a 
# HRRR file for the Alaska domain
#hrrr_file = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Pacific_Data/HRRR/hrrr.20140731/hrrr.t00z.wrfsfcf01.grib2'
#hrrr_file = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Atlantic_Data/HRRR/hrrr.20180831/hrrr.t00z.wrfsfcf01.grib2'
#hrrr_file = '../hrrr_20240219/conus/hrrr.t23z.wrfsfcf15.grib2'
hrrr_file = args.hrrrfile

# Create HRRR temporary file pathway
hrrr_tmp_nc = os.path.join(path,'HRRR_tmp.nc')
# Execute the wgrib2 command to convert the HRRR grib2 file into a netcdf file
############ Remember to set the Linux $WGRIB2 environmental variable to the wgrib2 exectuable pathway ###############
wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground|VGRD:10 m above ground|TMP:2 m above ground|SPFH:2 m above ground|APCP:surface|DSWRF:surface|DLWRF:surface|PRES:surface)" {hrrr_file} -netcdf {hrrr_tmp_nc}'
# Use subprocess module to execute the linux wgrib2 command and create the HRRR netcdf file
exitcode = subprocess.call(wgrib2_cmd, shell=True)

HRRR_proj = 'PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",38.5],PARAMETER["central_meridian",-97.5],PARAMETER["standard_parallel_1",38.5],PARAMETER["standard_parallel_2",38.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

met_data = xr.open_dataset(hrrr_tmp_nc)

geospatial_data = nc.Dataset(hrrr_tmp_nc)

met_data = met_data.rio.write_crs(HRRR_proj)

P_Enclosure = Polygon(read_polygon(polygon_enclosure))

dflowfm_df = gpd.GeoDataFrame({'DFLOWFM': ['POLYGON_ENCLOSURE']}, geometry=[P_Enclosure],crs="WGS84")

HRRR_x = met_data.x.values
HRRR_y = met_data.y.values

met_data.close()

x, y = np.meshgrid(HRRR_x,HRRR_y)

HRRR_lats = geospatial_data.variables['latitude'][:].data
HRRR_lons = (geospatial_data.variables['longitude'][:].data + 180) % 360 -180

geospatial_data.close()

HRRR_df = pd.DataFrame([])
HRRR_df['ids'] = np.arange(len(HRRR_lats.flatten()))
HRRR_df['x'] = HRRR_lons.flatten()
HRRR_df['y'] = HRRR_lats.flatten()
HRRR_df['HRRR_x'] = x.flatten()
HRRR_df['HRRR_y'] = y.flatten()


HRRR_gdf = gpd.GeoDataFrame(HRRR_df,geometry=gpd.points_from_xy(HRRR_df['x'],HRRR_df['y']),crs='WGS84')

intersection = HRRR_gdf.intersects(P_Enclosure)
ind_x = np.where(np.in1d(HRRR_x,HRRR_gdf.HRRR_x[intersection].values)==True)[0]
ind_y = np.where(np.in1d(HRRR_y,HRRR_gdf.HRRR_y[intersection].values)==True)[0]

clip_lats = HRRR_gdf[intersection].groupby(HRRR_gdf.HRRR_y).y.mean().values
clip_lons = HRRR_gdf[intersection].groupby(HRRR_gdf.HRRR_x).x.mean().values



#read in hgrid.cpp to get x,y,z in meters and list of elements
x = []
y = []
elem = []
with open(cppFile) as f:
    f.readline()
    line = f.readline()
    ne = int(line.split()[0])
    nn = int(line.split()[1])
    for i in range(nn):
        line = f.readline()
        x.append(float(line.split()[1]))
        y.append(float(line.split()[2]))
    for i in range(ne):
        line = f.readline()
        elem.append([int(line.split()[2]),int(line.split()[3]),int(line.split()[4])])
    
#calculate the area of each element and multiply by 1/density of water for use later
precip2flux = [Polygon(((x[e[0]-1],y[e[0]-1]),(x[e[1]-1],y[e[1]-1]),(x[e[2]-1],y[e[2]-1]))).area/1000 for e in elem]

#read in hgrid.ll to get x,y in degrees
x = []
y = []
with open(llFile) as f:
    f.readline()
    f.readline()
    for i in range(nn):
        line = f.readline()
        x.append(float(line.split()[1]))
        y.append(float(line.split()[2]))

#calculate the center of each triangle by finding the average lat,lon for each element        
avgX = [(x[e[0]-1]+x[e[1]-1]+x[e[2]-1])/3 for e in elem]
avgY = [(y[e[0]-1]+y[e[1]-1]+y[e[2]-1])/3 for e in elem]
propPoints = [[avgX[i],avgY[i]] for i in range(len(avgX))]

# Flatten HRRR clipped lons and lats
X = HRRR_lons[ind_y,:][:,ind_x]
Y = HRRR_lats[ind_y,:][:,ind_x]
x = X.flatten()
y = Y.flatten()

#create list of points in precipitation file
points = []
for i in range(len(x)):
    points.append([x[i],y[i]])

#perform delaunay triangularion on preciptiation points and file the triangles that we need precip data for    
t = Delaunay(points)
simplex = t.find_simplex(propPoints)
simplices = t.simplices[simplex,:]

area_lat = y[t.simplices[simplex,:]]
area_lon = x[t.simplices[simplex,:]]
arealatlon = [Polygon(((area_lon[i,0],area_lat[i,0]),(area_lon[i,1],area_lat[i,1]),(area_lon[i,2],area_lat[i,2]))).area for i in range(len(area_lat))]
    
area_cor = np.zeros((area_lat.shape[0],area_lat.shape[1]))
seq3 = [0,1,2,0,1]
for k in range(3):
    area_lat0 = [avgY,list(y[t.simplices[simplex,seq3[k+1]]].data),list(y[t.simplices[simplex,seq3[k+2]]].data)]
    area_lon0 = [avgX,list(x[t.simplices[simplex,seq3[k+1]]].data),list(x[t.simplices[simplex,seq3[k+2]]].data)]
    arealatlon0 = [Polygon(((area_lon0[0][i],area_lat0[0][i]),(area_lon0[1][i],area_lat0[1][i]),(area_lon0[2][i],area_lat0[2][i]))).area for i in range(len(area_lat))]
    area_cor[:,k] = list(map(truediv,arealatlon0,arealatlon))

if os.path.exists(outfile):
    os.remove(outfile)
    
# open a netCDF file to write
ncout = nc.Dataset(outfile,'w',format='NETCDF4')

# define axis size
ncout.createDimension('cols',3)
ncout.createDimension('elem',len(elem))
ncout.createDimension('nwmNodesX',X.shape[0])
ncout.createDimension('nwmNodesY',X.shape[1])

# create variables
ncp2f = ncout.createVariable('precip2flux','f8',('elem',))
ncsim = ncout.createVariable('simplex','int',('elem','cols',))
nccor = ncout.createVariable('area_cor','f8',('elem','cols',))
ncavgx = ncout.createVariable('avgX','f8',('elem',))
ncavgy = ncout.createVariable('avgY','f8',('elem',))
ncx = ncout.createVariable('x','f8',('nwmNodesX','nwmNodesY',))
ncy = ncout.createVariable('y','f8',('nwmNodesX','nwmNodesY',))

# copy axis from original dataset
ncp2f[:] = precip2flux
ncsim[:] = simplices
nccor[:] = area_cor
ncx[:] = X
ncy[:] = Y
ncavgx[:] = avgX
ncavgy[:] = avgY

ncout.close()

# Remove temporary HRRR netcdf file 
# created from wgrib2 command to clean
# up remaining unnecessary files
if os.path.exists(hrrr_tmp_nc):
    os.remove(hrrr_tmp_nc)
