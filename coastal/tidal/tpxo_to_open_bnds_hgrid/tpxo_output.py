#!/usr/bin/env python

from netCDF4 import Dataset

import numpy as np

import xarray as xr


dset = Dataset('/home/Zhengtao.Cui/nwm.v3.0.6_no_svn/parm/coastal/hawaii/open_bnds_hgrid.nc', 'r')

newnc = Dataset('lco_open_bnds_bhgrid.nc', 'w', format='NETCDF4_CLASSIC')

#Copy dimensions
print( dset.dimensions )
for the_dim in dset.dimensions.items():
     print( the_dim[0], the_dim[1].name, the_dim[1].size, len(the_dim) )
     if the_dim[1].name == 'nodeCount' or the_dim[1].name == 'openBndNodeCount':
        newnc.createDimension(the_dim[0], 8 if not the_dim[1].isunlimited() else None)
        print("xx")
     if the_dim[1].name == 'coordDim':
        newnc.createDimension(the_dim[0], 2 if not the_dim[1].isunlimited() else None)
        print("xxx")
        
#        nodeCount = 8695 ;
#        openBndNodeCount = 4350 ;
#        elementCount = 8695 ;
#        maxNodePElement = 3 ;
#        coordDim = 2 ;
#
#     newnc.createDimension(the_dim[0], the_dim[1].size if not the_dim[1].isunlimited() else None)
for name in dset.ncattrs():
     print( name )
     newnc.setncattr( name, dset.getncattr( name ) )

#       double nodeCoords(nodeCount, coordDim) ;
#                nodeCoords:units = "degrees" ;
#        int openBndNodes(openBndNodeCount) ;
#        int elementConn(elementCount, maxNodePElement) ;
#                elementConn:_FillValue = -1 ;
#                elementConn:long_name = "Node Indices that define the element connectivity" ;
#                elementConn:start_index = 0b ;
#        byte numElementConn(elementCount) ;
#                numElementConn:long_name = "Number of nodes per element" ;
#        double centerCoords(elementCount, coordDim) ;
#                centerCoords:units = "degrees" ;

# Copy variables
for v_name, varin in dset.variables.items():
        if v_name == "nodeCoords":
           outVar = newnc.createVariable(v_name, varin.datatype, ('nodeCount', 'coordDim', ) )
        if v_name == "openBndNodes":
           outVar = newnc.createVariable(v_name, varin.datatype, ('openBndNodeCount',) )

        if v_name == "nodeCoords" or v_name == "openBndNodes":
          print( 'v_name:', v_name, varin.datatype )
          print( 'shape=', varin.shape )
          print( 'shape= length', len(varin.shape) )

          print( outVar.datatype )
          print( 'outvar shape=', outVar.shape )
          print( 'outvar shape= length', len(outVar.shape) )

# Copy variable attributes
        if v_name == "nodeCoords" or v_name == "openBndNodes":
          for k in varin.ncattrs():
              print( k, '=', varin.getncattr(k) )
              outVar.setncattr( k, varin.getncattr(k) )

        if v_name == "openBndNodes":
           outVar[:] = [0,1,2,3,4,5,6,7]

        if v_name == "nodeCoords":
           outVar[:] = [[-96.22029,28.57183],
                        [-96.15505,28.58662],
                        [-96.07669,28.61743],
                        [-96.01858,28.61247],
                        [-95.99234,28.57649],
                        [-95.94361,28.59862],
                        [-95.92857,28.64843],
                        [-95.91665,28.69935] ]
           print( varin )

newnc.close()
dset.close()

