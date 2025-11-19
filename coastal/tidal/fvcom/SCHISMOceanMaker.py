###############################################################################
#  Module name: SCHISMOceanMaker                                              #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  12/24/2024                                          #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: Creates a SCHISM elev2D.th.cn file                            #
#                                                                             #
###############################################################################

import os
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import numpy as np
import netCDF4 as nc
import pandas as pd
from SCHISMGrid import SCHISMGrid
from FVCOMCollection import FVCOMCollection

class SCHISMOceanMaker:
        """
           Creates a SCHISM elev2D.th.cn file 
        """        

        def __init__(self, schism_grid ): 
            self.grid = schism_grid

        @property
        def grid(self):
            return self._grid

        @grid.setter
        def grid(self, g):
            self._grid = g

        def makeOcean(self, fvColet, outFile ):
           alltimes = fvColet.getTimes()
           units = fvColet.getRefTime()
           time_final = nc.num2date(alltimes,\
                   units=units,only_use_cftime_datetimes=False)
           for i in range(len(time_final)):
              if(time_final[i].minute >= 45):
                 time_final[i] = datetime(time_final[i].year,time_final[i].month,time_final[i].day,time_final[i].hour+1)
              else:
                 time_final[i] = datetime(time_final[i].year,time_final[i].month,time_final[i].day,time_final[i].hour)

           # Now we need to subset data based on user specified
           # start time and end time
           time_slice = pd.DataFrame([])
           time_slice['index_slice'] = np.arange(len(time_final))
           time_slice.index = pd.to_datetime(time_final)
           idx = (time_slice.index >= fvColet.starttime.strftime("%Y-%m-%d %H:%M:%S")) & (time_slice.index <= fvColet.endtime.strftime("%Y-%m-%d %H:%M:%S"))
           time_indices = time_slice.loc[idx,'index_slice'].values

           time_netcdf = time_slice.loc[idx,:].index -  pd.to_datetime(fvColet.starttime)
           time_netcdf = time_netcdf.total_seconds()

           #time_netcdf = nc.date2num(time_final,ref_time)

           schism_bnds = np.column_stack([self.grid.lons,self.grid.lats])
           FVCOM_points = np.column_stack([fvColet.getLons(),fvColet.getLats()])
           tree = cKDTree(FVCOM_points)
           _, open_bnds = tree.query(schism_bnds)

           zeta = fvColet.getZetas()
           zeta = zeta[:,open_bnds]
           zeta = zeta[time_indices,:]

           if os.path.exists(outFile):
               os.remove(outFile)

           # open a netCDF file to write
           ncout = nc.Dataset(outFile,'w',format='NETCDF4')

           # define axis size
           ncout.createDimension('time',None)
           ncout.createDimension('nOpenBndNodes',len(self.grid.lons))
           ncout.createDimension('nLevels',1)
           ncout.createDimension('nComponents',1)
           ncout.createDimension('one',1)

           # create time step variable
           nctstep = ncout.createVariable('time_step','f8',('one',)) 

           # create time axis
           nctime = ncout.createVariable('time','f8',('time',))
           #nctime.setncattr('units',ref_time)

           # create water level time series
           ncwl = ncout.createVariable('time_series','f8',('time','nOpenBndNodes','nLevels','nComponents',))

           # copy axis from original dataset
           nctstep[:] = FVCOMCollection.TIMESTEP_IN_SECS
           nctime[:] = time_netcdf
           ncwl[:] = zeta + self.grid.zeta0

           ncout.close()
