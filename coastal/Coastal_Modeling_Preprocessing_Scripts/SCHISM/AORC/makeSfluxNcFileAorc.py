#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 01:26:35 2022

@author: Camaron.George
"""
import os, numpy as np, netCDF4 as nc, pyproj as pj
from operator import truediv
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

#this script requires hgrid.utm, hgrid.ll, and netcdf precipitation file
#this path should include all of those files; if it doesn't, paths will need to be adjusted below
path = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Champlain/SCHISM/'
cppFile = path+'hgrid.cpp'
llFile = path+'hgrid.gr3'
atmoFile = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Lake_Ontario/AORC/AORC-OWP_2019051902z.nc4'
outfile = path + 'sflux2sourceInput.nc'

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

#convert NWM locations to lon/lat            
data = nc.Dataset(atmoFile)
X = data.variables['longitude'][:]
Y = data.variables['latitude'][:]
X,Y = np.meshgrid(X,Y)

I = np.where((X > np.min(x)-0.1) & (X < np.max(x)+0.1) & (Y < np.max(y)+0.1) & (Y > np.min(y)-0.1))

X = X[np.min(I[0]):np.max(I[0]),np.min(I[1]):np.max(I[1])]
Y = Y[np.min(I[0]):np.max(I[0]),np.min(I[1]):np.max(I[1])]

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

area_lat = y[simplices]
area_lon = x[simplices]
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
ncout.createDimension('arrayLocs',4)

# create variables
ncp2f = ncout.createVariable('precip2flux','f8',('elem',))
ncsim = ncout.createVariable('simplex','int',('elem','cols',))
nccor = ncout.createVariable('area_cor','f8',('elem','cols',))
ncavgx = ncout.createVariable('avgX','f8',('elem',))
ncavgy = ncout.createVariable('avgY','f8',('elem',))
ncx = ncout.createVariable('x','f8',('nwmNodesX','nwmNodesY',))
ncy = ncout.createVariable('y','f8',('nwmNodesX','nwmNodesY',))
nclocs = ncout.createVariable('minMax','int',('arrayLocs',))

# copy axis from original dataset
ncp2f[:] = precip2flux
ncsim[:] = simplices
nccor[:] = area_cor
ncx[:] = X
ncy[:] = Y
ncavgx[:] = avgX
ncavgy[:] = avgY
nclocs[:] = [np.min(I[0]),np.max(I[0]),np.min(I[1]),np.max(I[1])]

ncout.close()
