#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:43:00 2021

@author: Camaron.George@noaa.gov / rcabell@ucar.edu
"""

import glob
import numpy as np
import netCDF4 as nc
# from pyproj import CRS, Transformer
import os
from os import path

def round_down(n, decimals=0):
    import math
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def slp(temp, mixing, height, press):
    g0 = 9.80665
    Rd = 287.058
    epsilon = 0.622

    Tv = temp*(1 + (mixing/epsilon))/(1 + mixing)
    H = Rd * Tv / g0

    press_sl = press / np.exp(-height / H)

    return press_sl


class App:
    def __init__(self) -> None:
        # paths to where the forcing data is loaded / saved
        forcing_input_path = os.environ["COASTAL_FORCING_INPUT_DIR"]
        output_path = os.environ["COASTAL_WORK_DIR"]

        # file for precipitation only; will be used to combine with discharge later
        self.precip_out_nc = path.join(output_path, 'precip.nc')

        # file for rest of atmo variables, which will be used in the simulation so this should be saved
        # in the folder where it will be used in the run
        self.sflux_out_nc = path.join(output_path, 'sflux', 'sflux_air_1.0001.nc')

        # start times, sim length
        self.start_year = os.environ['FORCING_START_YEAR']
        self.start_month = os.environ['FORCING_START_MONTH']
        self.start_day = os.environ['FORCING_START_DAY']
        self.start_hour = int(os.environ['FORCING_START_HOUR'])
        self.length_hours = int(os.environ.get('LENGTH_HRS', 0))
        self.ana_flag = self.length_hours < 0
        self.length_hours = abs(self.length_hours)

        # # analysis file/data that will be used for zero time step (will be handled by FE)
        # file = path.join(forcing_input_path, 'nwm.t00z.analysis_assim.forcing.tm00.conus.latlon.nc')
        # if self.ana_flag:
        #     self.dir_date = os.environ["FORCING_END_DATE"]
        # else:
        #     self.dir_date = os.environ["FORCING_BEGIN_DATE"]
        # if len(self.dir_date) == 12:
        #     # remove minutes
        #     self.dir_date = self.dir_date[:-2]

        # TODO: compute actual file names to detect missing data
        self.forcing_input_glob = path.join(forcing_input_path, '*LDASIN_DOMAIN1')

        # Load geospatial data (lats, lons, terrain heights)
        geo_file = os.environ["GEOGRID_FILE"]
        with nc.Dataset(geo_file) as geo:
            self.height = geo['HGT_M'][0, :]
            self.lats = geo['XLAT_C'][0, :]
            self.lons = geo['XLONG_C'][0, :]

    def create_output(self):
        ncout = nc.Dataset(self.sflux_out_nc, 'w', format='NETCDF4')

        lat_bnds, lon_bnds = [28.34998744533768, 29.750635118300792], \
                             [-96.74116127039483, -95.31343778151957]


        # latitude lower and upper index
        #latli = np.argmin( np.abs( self.lats - lat_bnds[0] ) )
        #latui = np.argmin( np.abs( self.lats - lat_bnds[1] ) ) + 1

        #print( "latli = ", latli )
        #print( "latui = ", latui )

        # longitude lower and upper index
        #lonli = np.argmin( np.abs( self.lons - lon_bnds[0] ) )
        #lonui = np.argmin( np.abs( self.lons - lon_bnds[1] ) ) + 1

        #print( "lonli = ", lonli )
        #print( "lonui = ", lonui )

        lat_inds = np.where((self.lats > lat_bnds[0]) & (self.lats <= lat_bnds[1]))
        lon_inds = np.where((self.lons > lon_bnds[0]) & (self.lons <= lon_bnds[1]))

        lat_latli = np.min( lat_inds[0] )
        lat_latui = np.max( lat_inds[0] )

        lat_lonli = np.min( lat_inds[1] )
        lat_lonui = np.max( lat_inds[1] )

        lon_latli = np.min( lon_inds[0] )
        lon_latui = np.max( lon_inds[0] )

        lon_lonli = np.min( lon_inds[1] )
        lon_lonui = np.max( lon_inds[1] )

        #latli = lat_latli if lat_latli < lon_latli else lon_latli
        #latui = lat_latui if lat_latui > lon_latui else lon_latui
#
#        lonli = lon_lonli if lon_lonli < lat_lonli else lat_lonli
#        lonui = lon_lonui if lon_lonui > lat_lonui else lat_lonui
        latli = lat_latli
        latui = lat_latui + 1

        lonli = lon_lonli
        lonui = lon_lonui + 1

        print( lat_inds[0].shape, lat_inds[1].shape)

        files = sorted(glob.glob(self.forcing_input_glob))
        data = nc.Dataset(files[0], 'r')

        # get lon/lat locations of data
        # cf_dict = data['crs'].__dict__
        # proj_crs = CRS.from_cf(cf_dict)
        # xformer = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
        # x, y = np.meshgrid(data['x'][:], data['y'][:])
        # lons, lats = xformer.transform(x, y)

        ncout.createDimension('time', len(files)+1)
        #ncout.createDimension('ny_grid', self.lats.shape[0])
        #ncout.createDimension('nx_grid', self.lons.shape[1])
        ncout.createDimension('ny_grid', (latui-latli))

        #print( lat_inds[0].shape, lat_inds[1].shape)
        ncout.createDimension('nx_grid', lonui-lonli)

        nctime = ncout.createVariable('time', 'f4', ('time',))
        nclon = ncout.createVariable('lon', 'f4', ('ny_grid', 'nx_grid',))
        nclat = ncout.createVariable('lat', 'f4', ('ny_grid', 'nx_grid',))
        ncu = ncout.createVariable('uwind', 'f4', ('time', 'ny_grid', 'nx_grid',))
        ncv = ncout.createVariable('vwind', 'f4', ('time', 'ny_grid', 'nx_grid',))
        ncp = ncout.createVariable('prmsl', 'f4', ('time', 'ny_grid', 'nx_grid',))
        nct = ncout.createVariable('stmp', 'f4', ('time', 'ny_grid', 'nx_grid',))
        ncq = ncout.createVariable('spfh', 'f4', ('time', 'ny_grid', 'nx_grid',))
        ncrain = ncout.createVariable('rain', 'f4', ('time', 'ny_grid', 'nx_grid',))

        nctime.long_name = "Time"
        nctime.standard_name = "time"

        time = np.arange(0, (1/24)*(len(files)+1), 1/24)
        time += int(self.start_hour) / 24.0
        time[0] = round_down(time[0], 7)
        nctime[:] = time

        data.close()

        # set up field variables

        baseDateStr = f"{self.start_year}-{self.start_month}-{self.start_day}"           # "2020-08-26 0'
        baseDate = list(map(np.int32, [self.start_year, self.start_month, self.start_day, 0]))
        nctime.units = "days since "+baseDateStr
        nctime.base_date = baseDate


        nclon.long_name = "Longitude"
        nclon.standard_name = "longitude"
        nclon.units = "degrees_east"
        print("nclon shape = ", nclon.shape)
        print("self lons shape =", self.lons.shape)
        nclon[:] = self.lons[latli:latui, lonli:lonui]

        nclat.long_name = "Latitude"
        nclat.standard_name = "latitude"
        nclat.units = "degrees_north"
        nclat[:] = self.lats[latli:latui, lonli:lonui]

        ncu.long_name = "Surface Eastward Air Velocity (10m AGL)"
        ncu.standard_name = "eastward_wind"
        ncu.units = "m/s"

        ncv.long_name = "Surface Northward Air Velocity (10m AGL)"
        ncv.standard_name = "northward_wind"
        ncv.units = "m/s"

        ncp.long_name = "Pressure reduced to MSL"
        ncp.standard_name = "air_pressure_at_sea_level"
        ncp.units = "Pa"

        nct.long_name = "Surface Air Temperature (2m AGL)"
        nct.standard_name = "air_temperature"
        nct.units = "K"

        ncq.long_name = "Surface Specific Humidity (2m AGL)"
        ncq.standard_name = "specific_humidity"
        ncq.units = "kg/kg"

        ncrain.long_name = "Surface Precipitation Rate"
        ncrain.standard_name = "precipitation_flux"
        ncrain.units = "mm s^-1"

        for i, file in enumerate(files):
            print( file )
            data = nc.Dataset(file)

            nct[i, :] = data.variables['T2D'][:, latli:latui, lonli:lonui]
            ncq[i, :] = data.variables['Q2D'][:, latli:latui, lonli:lonui]
            ncu[i, :] = data.variables['U2D'][:, latli:latui, lonli:lonui]
            ncv[i, :] = data.variables['V2D'][:, latli:latui, lonli:lonui]
            ncp[i, :] = slp(nct[i, :], ncq[i, :], self.height[latli:latui, lonli:lonui], \
                    data.variables['PSFC'][:, latli:latui, lonli:lonui])
            ncrain[i, :] = data.variables['RAINRATE'][:, latli:latui, lonli:lonui]

            data.close()

        # copy last data
        ncu[-1] = ncu[-2]
        ncv[-1] = ncv[-2]
        ncp[-1] = ncp[-2]
        nct[-1] = nct[-2]
        ncq[-1] = ncq[-2]
        ncrain[-1] = ncrain[-2]

        ncout.close()


if __name__ == '__main__':
    App().create_output()
