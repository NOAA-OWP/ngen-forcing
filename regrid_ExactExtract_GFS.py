import numpy as np
import geopandas as gpd
import netCDF4 as nc4
import os
import glob
from pathlib import Path
from os.path import join
import argparse
import math
from math import tau as TWO_PI
import pandas as pd
from  multiprocessing import Process, Lock
import multiprocessing
import time
import datetime
import re
# load python C++ binds from ExactExtract module library
from exactextract import GDALDatasetWrapper, GDALRasterWrapper, Operation, MapWriter, FeatureSequentialProcessor, GDALWriter
# must import gdal to properly read and partiton rasters
# from AORC netcdf files
from osgeo import gdal

# Import Beta version of pywgrib2_s to directly manipulate GFS grib2 files
import pywgrib2_s

def get_date_time_csv(path):
    """
    Extract the date-time from the file path
    """
    path = Path(path)
    name = path.stem
    date_time = name.split('.')[0]
    date_time = date_time.split('_')[2]  #this index may depend on the naming format of the forcing data
    date_time = re.sub('\D','',date_time)
    return date_time

def create_ngen_netcdf(weighted_csv_files,gfs_ncfile):
    """
    Create NextGen netcdf file with specified format
    """

    # Initalize GFS data arrays to exact csv file data 
    cat_id = np.zeros(num_catchments, dtype="S16") # "cat-" + up to 7 digits... but maybe plus \0? Add some padding, JIC.
    time = np.zeros((num_catchments,num_files), dtype=float)

    PRATE_surface = np.zeros((num_catchments,num_files), dtype=float)
    DLWRF_surface = np.zeros((num_catchments,num_files), dtype=float)
    DSWRF_surface = np.zeros((num_catchments,num_files), dtype=float)
    PRES_surface = np.zeros((num_catchments,num_files), dtype=float)
    SPFH_2maboveground = np.zeros((num_catchments,num_files), dtype=float)
    TMP_2maboveground = np.zeros((num_catchments,num_files), dtype=float)
    UGRD_10maboveground = np.zeros((num_catchments,num_files), dtype=float)
    VGRD_10maboveground = np.zeros((num_catchments,num_files), dtype=float)


    # get catchment ids and reference start time for NextGen forcing file
    df = pd.read_csv(weighted_csv_files[0])
    cat_id[:] = df['cat-id'].values
    start_time = get_date_time_csv(weighted_csv_files[0])

    # Loop over all csv files
    for i in np.arange(len(weighted_csv_files)):
        # Get datetime of file, add to time variable
        file_date = get_date_time_csv(weighted_csv_files[i])
        time[:,i] = (pd.Timestamp(datetime.datetime.strptime(file_date,'%Y%m%d%H')) - ref_date).total_seconds() # Broadcast at work
        df = pd.read_csv(weighted_csv_files[i])
        
        # Add the ExactExtract csv data to netcdf variables
        PRATE_surface[:,i] = df['PRATE_surface'].values
        DLWRF_surface[:,i] = df['DLWRF_surface'].values
        DSWRF_surface[:,i] = df['DSWRF_surface'].values
        PRES_surface[:,i] = df['PRES_surface'].values
        SPFH_2maboveground[:,i] = df['SPFH_2maboveground'].values
        TMP_2maboveground[:,i] = df['TMP_2maboveground'].values
        UGRD_10maboveground[:,i] = df['UGRD_10maboveground'].values
        VGRD_10maboveground[:,i] = df['VGRD_10maboveground'].values

    # first read GFS metadata in to save to NextGen forcing file
    ds = nc4.Dataset(gfs_ncfile)

    #create output netcdf file name
    output_path = join(forcing, "NextGen_GFS_forcing_"+start_time+".nc")

    #make the data set
    filename = output_path
    filename_out = output_path

     
    # write data to netcdf files
    ncfile_out = nc4.Dataset(filename_out, 'w', format='NETCDF4')


    #add the dimensions
    time_dim = ncfile_out.createDimension('time', None)
    catchment_id_dim = ncfile_out.createDimension('catchment-id', num_catchments)
    string_dim =ncfile_out.createDimension('str_dim', 1)

    # create variables
    cat_id_out = ncfile_out.createVariable('ids', 'str', ('catchment-id'), fill_value="None")
    time_out = ncfile_out.createVariable('Time', 'double', ('catchment-id','time',), fill_value=-99999)
    PRATE_surface_out = ncfile_out.createVariable('RAINRATE', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    TMP_2maboveground_out = ncfile_out.createVariable('T2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    SPFH_2maboveground_out = ncfile_out.createVariable('Q2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    UGRD_10maboveground_out = ncfile_out.createVariable('U2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    VGRD_10maboveground_out = ncfile_out.createVariable('V2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    PRES_surface_out = ncfile_out.createVariable('PSFC', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    DSWRF_surface_out = ncfile_out.createVariable('SWDOWN', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    DLWRF_surface_out = ncfile_out.createVariable('LWDOWN', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)

    #set output netcdf file atributes
    varout_dict = {'time':time_out,
                   'PRATE_surface':PRATE_surface_out, 'DLWRF_surface':DLWRF_surface_out, 'DSWRF_surface':DSWRF_surface_out,
                   'PRES_surface':PRES_surface_out, 'SPFH_2maboveground':SPFH_2maboveground_out, 'TMP_2maboveground':TMP_2maboveground_out,
                   'UGRD_10maboveground':UGRD_10maboveground_out, 'VGRD_10maboveground':VGRD_10maboveground_out}

    #copy all attributes from input netcdf file
    for name, variable in ds.variables.items():
        if name == 'latitude' or name == 'longitude':
            pass
        else:
            if(name != 'HGT_surface'):
                varout_name = varout_dict[name]
                for attrname in variable.ncattrs():
                    if name == "time" and attrname == "units":
                        #slight hack here to be compatible with current NetCDFPerFeatureDataProvider
                        # ... change it instead?
                        #see also https://www.unidata.ucar.edu/software/netcdf/time/recs.html
                        setattr(varout_name, "units", "seconds")
                        setattr(varout_name, "epoch_start", "01/01/1970 00:00:00")
                    elif attrname != '_FillValue':
                        setattr(varout_name, attrname, getattr(variable, attrname))

    #drop the scale_factor and add_offset from the output netcdf forcing file attributes
    for key, varout_name in varout_dict.items():
        if key != 'time':
            try:
                del varout_name.scale_factor
            except: 
                print("No scale factor in forcing files. No keys to tweak for output netcdf")
            try:
                del varout_name.add_offset
            except:
                print("No add offset in forcing files. No keys to tweak for output netcdf")


    #####################################################################

    #set attributes for additional variables
    setattr(cat_id_out, 'description', 'catchment_id')
   
    cat_id_out[:] = cat_id[:]
    time_out[:,:] = time[:,:]
    PRATE_surface_out[:,:] = PRATE_surface[:,:]
    DLWRF_surface_out[:,:] = DLWRF_surface[:,:]
    DSWRF_surface_out[:,:] = DSWRF_surface[:,:]
    PRES_surface_out[:,:] = PRES_surface[:,:]
    SPFH_2maboveground_out[:,:] = SPFH_2maboveground[:,:]
    TMP_2maboveground_out[:,:] = TMP_2maboveground[:,:]
    UGRD_10maboveground_out[:,:] = UGRD_10maboveground[:,:]
    VGRD_10maboveground_out[:,:] = VGRD_10maboveground[:,:]

    # Now close NextGen netcdf file
    # and AORC file
    ncfile_out.close()
    ds.close()


def GFS_bias_correction(dataset, hour, current_output_step):

    # Perform bias correction for incoming short wave radiation
    dataset['DSWRF_surface'] = gfs_lwdown_bias_correction(dataset['DSWRF_surface'].values, hour, current_output_step)

   
    # Perform bias correction for 2m Air Temperature
    dataset['TMP_2maboveground'] = gfs_tmp_bias_correction(dataset['TMP_2maboveground'].values, hour, current_output_step)

    # Perform bias correction for u and v wind components collectively
    dataset['UGRD_10maboveground'], dataset['VGRD_10maboveground'] = gfs_wspd_bias_correction(dataset['UGRD_10maboveground'].values, dataset['VGRD_10maboveground'].values, hour, current_output_step)
  
    return dataset
    
# GFS temperature bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def gfs_tmp_bias_correction(tmp_data, hour, current_output_step):
    hh = hour
    net_bias_mr = -0.18
    fhr_mult_mr = 0.002
    diurnal_ampl_mr = -1.4
    diurnal_offs_mr = -2.1

    fhr = current_output_step

    bias_corr = net_bias_mr + fhr_mult_mr * fhr + diurnal_ampl_mr * math.sin(diurnal_offs_mr + hh / 24 * TWO_PI)
    tmp_data = tmp_data + bias_corr

    return tmp_data

# GFS Incoming longwave radiation bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def gfs_lwdown_bias_correction(lwdown_data, hour, current_output_step):

    hh = hour
    fhr = current_output_step

    lwdown_net_bias_mr = 9.9
    lwdown_fhr_mult_mr = 0.00
    lwdown_diurnal_ampl_mr = -1.5
    lwdown_diurnal_offs_mr = 2.8

    bias_corr = lwdown_net_bias_mr + lwdown_fhr_mult_mr * fhr + lwdown_diurnal_ampl_mr * \
                math.sin(lwdown_diurnal_offs_mr + hh / 24 * TWO_PI)

    lwdown_data = lwdown_data + bias_corr

    return lwdown_data

# GFS wind speed bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def gfs_wspd_bias_correction(ugrd, vgrd, hour, current_output_step):

    hh = hour
    fhr = current_output_step
    wspd_net_bias_mr = -0.20
    wspd_fhr_mult_mr = 0.00
    wspd_diurnal_ampl_mr = -0.32
    wspd_diurnal_offs_mr = -1.1

    # need to get wind speed from U, V components
    wdir = np.arctan2(vgrd, ugrd)
    wspd = np.sqrt(np.square(ugrd) + np.square(vgrd))

    bias_corr = wspd_net_bias_mr + wspd_fhr_mult_mr * fhr + \
                wspd_diurnal_ampl_mr * math.sin(wspd_diurnal_offs_mr + hour / 24 * TWO_PI)

    wspd = wspd + bias_corr
    wspd = np.where(wspd < 0, 0, wspd)
    # Now seperate wind speed components and return back 
    ugrid_out = wspd * np.cos(wdir)
    vgrid_out = wspd * np.sin(wdir)

    return ugrid_out, vgrid_out


    
def GFS_downscaling(csv_results,timestamp):
    
    # Extract regridded GFS model elevation to use as
    # input in the GFS downscaling functions
    elev = csv_results['HGT_surface'].values

    # Get hour of day from pandas timestamp
    hour = timestamp.hour[0]

    # Perform downscaling on GFS temperature and reassign 
    # data to pandas dataframe
    tmp_2m = csv_results['TMP_2maboveground'].values
    tmp_downscaled = gfs_temp_downscaling_simple_lapse(tmp_2m, elev)
    csv_results['TMP_2maboveground'] = tmp_downscaled

    # Perform downscaling on GFS surface pressure and reassign
    # data to pandas dataframe
    pres_old = csv_results['PRES_surface'].values
    pres_downscaled = gfs_pres_downscaling_classic(pres_old,tmp_downscaled,elev)
    csv_results['PRES_surface'] = pres_downscaled

    # Perform downscaling on GFS 2m specific humidity and reassign
    # data to pandas dataframe
    tmpHumidity = csv_results['SPFH_2maboveground'].values
    humidity_downscaled = gfs_hum_downscaling_classic(tmpHumidity,tmp_downscaled,tmp_2m,pres_old)
    csv_results['SPFH_2maboveground'] = humidity_downscaled

    
    # Perform downscaling on GFS incoming short wave radiation and reassign
    # data to pandas dataframe
    dswr = csv_results['DSWRF_surface'].values
    dswr_downscaled = gfs_dswr_downscaling_topo_adj(dswr, timestamp)
    csv_results['DSWRF_surface'] = dswr_downscaled

    return csv_results

# GFS Temperature downscaling function based on simple lapse rate derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def gfs_temp_downscaling_simple_lapse(tmp_in, elev):
    """
    Function that applies a single lapse rate adjustment to modeled
    2-meter temperature by taking the difference of the native
    input elevation and the WRF-hydro elevation.
    :param tmp_in:
    :param elev:
    :param NextGen_catchment_features: globally available
    :return: air_pressure: downscaled surface air pressure
    """

    tmp2m = tmp_in

    elevDiff = elev - NextGen_catchment_features.height.values

    # Apply single lapse rate value to the input 2-meter
    # temperature values.
    tmp2m = tmp2m + (6.49/1000.0)*elevDiff

    return tmp2m

# GFS pressure downscaling function based on hypsometirc equation derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def gfs_pres_downscaling_classic(pres_in, tmp2m, elev):
    """
    Generic function to downscale surface pressure to the WRF-Hydro domain.
    :param : pres_in
    :param : tmp2m -- temperature
    :param : elev -- GFS elevation grid
    :param NextGen_catchment_features:  globally available
    :return: pres_downscaled - downscaled surface air pressure
    """
    elevDiff = elev - NextGen_catchment_features.height.values

    pres_downscaled = pres_in + (pres_in*elevDiff*9.8)/(tmp2m*287.05)

    return  pres_downscaled

# GFS specific humidity downscaling function based on hypsometirc equation derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def gfs_hum_downscaling_classic(tmpHumidity, tmp2m, t2dTmp, psfcTmp):
    """
    Function for downscaling 2-meter specific humidity using already downscaled
    2-meter temperature, unadjusted surface pressure, and downscaled surface
    pressure.
    :param humidity, --- 2-meter specific humidity
           tmp2m -- downscaled temperatures ,
           t2dTmp -- orignial undownscaled temperatures
           psfcTmp--- orignial undownscaled air pressure
    :return: q2Tmp -- downscaled 2-meter specific humidity
    """

    # First calculate relative humidity given original surface pressure and 2-meter
    # temperature

    tmpHumidity = tmpHumidity/(1-tmpHumidity)

    T0 = 273.15
    EP = 0.622
    ONEMEP = 0.378
    ES0 = 6.11
    A = 17.269
    B = 35.86

    EST = ES0 * np.exp((A * (t2dTmp - T0)) / (t2dTmp - B))
    QST = (EP * EST) / ((psfcTmp * 0.01) - ONEMEP * EST)
    relHum = 100 * (tmpHumidity / QST)

    term1 = A * (tmp2m - T0)
    term2 = tmp2m - B
    EST = np.exp(term1 / term2) * ES0

    QST = (EP * EST) / ((tmpHumidity/100.0) - ONEMEP * EST)
    q2Tmp = QST * (relHum * 0.01)
    q2Tmp = q2Tmp / (1.0 + q2Tmp)

    return q2Tmp

# GFS incoming short wave radiation downscaling function based on topographic adjustment derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def gfs_dswr_downscaling_topo_adj(dswr_in, timestamp):
    """
    Topographic adjustment of incoming shortwave radiation fluxes,
    given input parameters.
    :param dswr_in:
    :param current_output_date:
    :param NextGen_catchment_features: global variable
    :return: SWDOWN_OUT:  downscaled incoming shortwave radiation
    """

    # By the time this function has been called, necessary input static grids (height, slope, etc),
    # should have been calculated for each local slab of data.
    DEGRAD = math.pi/180.0
    DPD = 360.0/365.0

    # For short wave radiation
    DECLIN = 0.0
    SOLCON = 0.0

    coszen_loc, hrang_loc = calc_coszen(DECLIN,timestamp)
    dswr_downscaled = TOPO_RAD_ADJ_DRVR(dswr_in,coszen_loc,DECLIN,SOLCON,hrang_loc)
    return dswr_downscaled

# Calculate cosine of solar zenith angle based on time of day derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def calc_coszen(declin,timestamp):
    """
    Downscaling function to compute radiation terms based on current datetime
    information and lat/lon grids.
    :param input_forcings:
    :param declin:
    :return:
    """
    degrad = math.pi / 180.0
    gmt = 0

    # Calculate the current julian day.
    julian = timestamp.to_julian_date()[0]

    da = 6.2831853071795862 * ((julian - 1) / 365.0)
    eot = ((0.000075 + 0.001868 * math.cos(da)) - (0.032077 * math.sin(da)) - \
           (0.014615 * math.cos(2 * da)) - (0.04089 * math.sin(2 * da))) * 229.18
    xtime = timestamp.hour[0] * 60.0  # Minutes of day
    xt24 = int(xtime) % 1440 + eot
    tloctm = NextGen_catchment_features.longitude_grid.values/15.0 + gmt + xt24/60.0
    hrang = ((tloctm - 12.0) * degrad) * 15.0
    xxlat = NextGen_catchment_features.latitude_grid.values * degrad
    coszen = np.sin(xxlat) * math.sin(declin) + np.cos(xxlat) * math.cos(declin) * np.cos(hrang)

    # Reset temporary variables to free up memory.
    tloctm = None
    xxlat = None

    return coszen, hrang

# GFS topography shortwave radiation adjustment based on time of day derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py    
def TOPO_RAD_ADJ_DRVR(dswr_in,COSZEN,declin,solcon,hrang2d):
    """
    Downscaling driver for correcting incoming shortwave radiation fluxes from a low
    resolution to a a higher resolution.
    :param NextGen_catchment_features:
    :param dswr_in
    :param COSZEN:
    :param declin:
    :param solcon:
    :param hrang2d:
    :return:
    """
    degrad = math.pi / 180.0

    ny = len(NextGen_catchment_features.ny_local.values)
    nx = len(NextGen_catchment_features.nx_local.values)

    xxlat = NextGen_catchment_features.latitude_grid.values*degrad

    # Sanity checking on incoming shortwave grid.
    SWDOWN = dswr_in
    SWDOWN[np.where(SWDOWN < 0.0)] = 0.0
    SWDOWN[np.where(SWDOWN >= 1400.0)] = 1400.0
    COSZEN[np.where(COSZEN < 1E-4)] = 1E-4

    corr_frac = np.empty([ny], np.int64)
    diffuse_frac = np.empty([ny], np.int64)
    corr_frac[:] = 0
    diffuse_frac[:] = 0

    indTmp = np.where((NextGen_catchment_features.slope.values == 0.0) &
                      (SWDOWN <= 10.0))
    corr_frac[indTmp] = 1

    term1 = np.sin(xxlat) * np.cos(hrang2d)
    term2 = ((0 - np.cos(NextGen_catchment_features.slp_azi.values)) *
             np.sin(NextGen_catchment_features.slope.values))
    term3 = np.sin(hrang2d) * (np.sin(NextGen_catchment_features.slp_azi.values) *
                               np.sin(NextGen_catchment_features.slope.values))
    term4 = (np.cos(xxlat) * np.cos(hrang2d)) * np.cos(NextGen_catchment_features.slope.values)
    term5 = np.cos(xxlat) * (np.cos(NextGen_catchment_features.slp_azi.values) *
                             np.sin(NextGen_catchment_features.slope.values))
    term6 = np.sin(xxlat) * np.cos(NextGen_catchment_features.slope.values)

    csza_slp = (term1 * term2 - term3 + term4) * math.cos(declin) + \
               (term5 + term6) * math.sin(declin)

    # Topographic shading
    csza_slp[np.where(csza_slp <= 1E-4)] = 1E-4

    # Correction factor for sloping topographic: the diffuse fraction of solar
    # radiation is assumed to be unaffected by the slope.
    corr_fac = diffuse_frac + ((1 - diffuse_frac) * csza_slp) / COSZEN
    corr_fac[np.where(corr_fac > 1.3)] = 1.3

    # Peform downscaling
    SWDOWN_OUT = SWDOWN * corr_fac

    # Reset variables to free up memory
    # corr_frac = None
    diffuse_frac = None
    term1 = None
    term2 = None
    term3 = None
    term4 = None
    term5 = None
    term6 = None

    return SWDOWN_OUT

###### The python ExactExtract method will be improved #####
###### once we've resolved GFS data issues and we may #####
###### just generate weights and perform manual regrid ####
###### method instead of this approach since we can   #####
###### speed up calculations of GFS aerial weighting #####
##### averages                                       #####
def python_ExactExtract(gfs_grib2_file,file_index_id):

    # Initalize pandas dataframe to save results to csv file
    csv_results = pd.DataFrame([])

    # Initalize which raster bands we will will extract
    # in the grib2 file. For the initalized GFS forecast 
    # (hour 0)however the band indices will differ, so
    # we are accounting for that. Were also accounting
    # for the fact that at hour 0, the initalized GFS
    # model has no data available for PRATE
    if(file_index_id == 0):
        gfs_band_rasters = [6,7,28,29,30,31,0,48,49]
    else:
        gfs_band_rasters = [6,7,33,34,39,40,43,95,96]

    # Initalize gdal gtiff driver before GFS variable loop
    driver = gdal.GetDriverByName("GTiff")

    # Open GFS grib2 dataset using GDAL library
    # before the GFS variable loop
    dataset = gdal.Open(gfs_grib2_file)

    # Grab the first raster band from GFS grib2 file
    # to obtain GFS metadata
    band = dataset.GetRasterBand(1)

    # Grab the first GFS raster band in the dataset and
    # extract metadata into array to get datetime info
    metadata = np.array(list(band.GetMetadata().items()))

    # Extract the metadata reference and forecast time
    # to create current GFS forecast timestamp, this will be
    # needed to disscet within the GFS bias corrections and
    # downscaling functions
    timestamp = pd.to_datetime(metadata[4][1].split(' ')[5].split('=')[1][:-1]) + pd.TimedeltaIndex([float(metadata[3][1].split(' ')[0])],'s')

    # Create date time string for csv files
    date_time = str(timestamp[0]).replace("-","").replace(" ","").split(":")[0]

    # We are only saving csv files currentlya as diagnostics for
    # evaluating ExactExtract python module performace
    NextGen_csv = join(exactextract_files,'NextGen_forcings_'+str(date_time)+'.csv')

    # get datetime of forcing file for global time variable from netcdf file
    # which is in seconds since AORC reference date and the date-time string
    time_final = (timestamp - ref_date).total_seconds()

    # get current hour of GFS forecast cycle in metadata, 
    # this is needed within GFS bias correction and downscaling functions
    current_output_step = float(metadata[3][1].split(' ')[0])/3600.0

    # get hour of the day based on current GFS timestamp, this is needed
    # within the GFS bias correction and downscaling functions
    hour = timestamp.hour[0]

    # loop over each meteorological variable and call 
    # ExactExtract to regrid raster to lumped sum for
    # a given NextGen catchment
    for i in np.arange(len(GFS_met_vars)):

        # Define which variable we are looking
        # to regrid for GFS forcings
        variable = GFS_met_vars[i]

        # Assign GFS band to extract grib2 data from
        # based on the variable assigned in the loop
        gfs_band = gfs_band_rasters[i]

        # Since GFS forecast files are sorted, index 0 implies the initalized
        # forecast time for the model run. NWP models won't generate time=0
        # flux fields, the model physics need to integrate forward in order
        # to calculate fluxes. Therefore, for now we must set precipitation
        # totals to zero. Skip the ExactExtract process and move to next variable
        if(file_index_id == 0 and variable == 'PRATE_surface'):
            csv_results[variable] = np.zeros(len(results['cat-id']))
        else:

            # Define gdal writer to only return ExactExtract
            # regrid results as a python dict
            writer = MapWriter()

            # Initalize GFS tif file were creating for the transformed
            # CONUS subset of a GFS raster
            gfs_tif = join(forcing, "GFS_"+str(file_index_id)+".tif")
    
            # Get GFS data pertaining to variable in the loop
            band = dataset.GetRasterBand(gfs_band)
            # Get CONUS susbet of the GFS raster, where grid spacing is even
            # and NextGen data of interest. Flag for unit conversion for temperature.
            if(variable == "TMP_2maboveground"):
                # Convert GFS grib2 temp data from Celsius to Kelvin
                data = band.ReadAsArray(i1_conus,j1_conus,cols_conus,rows_conus) + 273.15 
            else:
                data = band.ReadAsArray(i1_conus,j1_conus,cols_conus,rows_conus)
 
            # Now create a geotiff dataset of the CONUS GFS raster subset
            dst_ds = driver.Create(gfs_tif,cols_conus,rows_conus,1,gdal.GDT_Float32)
            # Create new raster band for geotiff
            new_band = dst_ds.GetRasterBand(1)
            # Set raster band GFS variable name
            new_band.SetDescription(variable)
            # Write data to geotiff raster
            new_band.WriteArray(data)
            # Set CONUS geometry and GFS grid spacing
            dst_ds.SetGeoTransform(transform)
            # Set GFS projection to geotiff (EPSG 9122)
            dst_ds.SetProjection(projection.ExportToWkt())
            # Close geotiff file for use in ExactExtract
            dst_ds = None

            # Define NextGen hydrofabric dataset 
            dsw = GDALDatasetWrapper(hyfabfile)

            # Define raster wrapper for AORC meteorological variable
            # and specify nc_file attribute to be True. Otherwise,
            # this function will expect a .tif file
            rsw = GDALRasterWrapper(gfs_tif) 


            
            # Define operation (mean) to use for raster
            op = Operation.from_descriptor('mean('+variable+')', raster=rsw)
       
            # Process the data and write results to writer instance
            processor = FeatureSequentialProcessor(dsw, writer, [op])
            processor.process()

            # convert dict results to pandas dataframe 
            results = pd.DataFrame(writer.output.items(),columns=['cat-id',variable])

            # find indices where scale factor is for AORC variable
            idx = np.where(variable==GFS_met_vars)[0][0]


            # save results to global GFS variables and account for scale factor
            # and offset of variable (if any). Set flag to just append 'cat-id'
            # data to just first variable in the loop to pandas dataframe
            if(i == 0):
                csv_results['cat-id'] = results['cat-id'].values
                csv_results[variable] = np.stack(results[variable].values,axis=0).flatten()*scale_factor[idx] + add_offset[idx]
            else:
                csv_results[variable] = np.stack(results[variable].values,axis=0).flatten()*scale_factor[idx] + add_offset[idx]


            # Since we have finished using the GFS netcdf file
            # we can now just remove file from system and save
            # disk storage on server
            os.remove(gfs_tif)

            # Flush changes to disk
            writer = None
    ################ Once GFS regridding is complete, we now call ##############
    ################ the bias correcation and downscaling functions ###########
    ################ sequentially according to NCAR documentation ############

    ############### This is where we call GFS bias correction function  #########

    csv_results = GFS_bias_correction(csv_results,hour,current_output_step)

    ############### This is where we call GFS downscaling function  #########

    csv_results = GFS_downscaling(csv_results,timestamp)

    ####################################################################


    ####### We will eventually just return the data ######
    ####### to  the thread and dynamically use it  #######

    # Save pandas dataframe of AORC ExactExtract
    # regridded data to csv file for diagnostics
    csv_results.to_csv(NextGen_csv,index=False)



def gdal_grib2_transformation(grib2_file):
    # Open grib2 dataset using gdal module
    dataset = gdal.Open(grib2_file)

    # Get just the first raster band of dataset to
    # transform and subset the grid
    band = dataset.GetRasterBand(1)

    # Get rows and columns of raster dataset
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    # Get current GFS grib2 transformation of dataset
    transform = dataset.GetGeoTransform()

    # Get min and max values of latitude and longitude coordinates
    minx = transform[0]
    maxx = transform[0] + cols * transform[1] + rows * transform[2]

    miny = transform[3] + cols * transform[4] + rows * transform[5]
    maxy = transform[3]

    # Get width and height dimsenions of grid spacing (degrees)
    width = maxx - minx
    height = maxy - miny

    # Get origin coordinates of GFS grid and its grid spacing
    # in the x and y directions (not consistent as you near the poles)
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Pre-define CONUS subset 
    newminx = 235.0
    newmaxx = 294.0
    newminy = 24.0
    newmaxy = 52.0

    # Define coordinate bounds for CONUS
    p1 = (newminx, newmaxy)
    p2 = (newmaxx, newminy)

    # find new array indices where CONUS
    # domain is bounded for grib data
    i1 = int((p1[0] - xOrigin) / pixelWidth)
    j1 = int((yOrigin - p1[1])  / pixelHeight)
    i2 = int((p2[0] - xOrigin) / pixelWidth)
    j2 = int((yOrigin - p2[1]) / pixelHeight)

    # now redefine the new number of columns and rows
    # for the subset CONUS domain
    new_cols = i2 - i1
    new_rows = j2 - j1

    # Now lets define new coordinate transformation bounds
    # for the raster arrays in the CONUS domain.
    # We also convert longitude coordinates from 0-360
    # to -180 180 in order to properly construct the raster
    # tif file for gdal to read in ExactExtract
    new_x = ((xOrigin + i1*pixelWidth) + 180) % 360 - 180
    new_y = yOrigin - j1*pixelHeight

    # Define new transformation for CONUS tif files
    new_transform = (new_x, transform[1], transform[2], new_y, transform[4], transform[5])

    # Get the original Projection of the grib2 file
    # to save the projection data to tif files
    wkt = dataset.GetProjection()
    srs = gdal.osr.SpatialReference()
    srs.ImportFromWkt(wkt)

    return new_transform, srs, i1, j1, new_cols, new_rows


def process_sublist(data : dict, lock: Lock, num: int):
    num_files = len(data["forcing_files"])    

    for i in range(num_files):
        # extract forcing file and file index
        gfs_grib_file = data["forcing_files"][i]
        file_index_id = data["file_index"][i]
     
        # Call python ExactExtract routine to directly extract 
        # GFS regridded results to global GFS variables
        python_ExactExtract(gfs_grib_file,file_index_id)
     

if __name__ == '__main__':
   
    #example: python regrid_ExactExtract_GFS.py -i /apd_common/test/test_data/forcing_script -o /apd_common/test/test_data/forcing_script -f forcing_files/ -e_csv csv_files/ -g /apd_common/test/test_data/forcing_script/GFS_grib2 -c /apd_common/test/test_data/forcing_script/hydrofabric/catchment_data_test.geojson -c_feat /apd_common/test/test_data/forcing_script/hydrofabric/NextGen_catchment_features.csv -j 12

    #parse the input and output root directory
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_root", type=str, required=True, help="The input directory with csv files")
    parser.add_argument("-o", dest="output_root", type=str, required=True, help="The output file path")
    parser.add_argument("-g", dest="gfs_grib2", type=str, required=True, help="The input gfs grib2 files directory")
    parser.add_argument("-f", dest="forcing", type=str, required=True, help="The output forcing files sub_dir")
    parser.add_argument("-e_csv", dest="ExactExtract_csv_files", type=str, required=True, help="The output sub_dir for ExactExtract csv files created from each GFS file")
    parser.add_argument("-c", dest="catchment_source", type=str, required=True, help="The hydrofabric catchment file or data ")
    parser.add_argument("-c_feat", dest="catchment_features", type=str, required=True, help="The hydrofabric catchmment characteristic csv file ")
    parser.add_argument("-j", dest="num_processes", type=int, required=False, default=96, help="The number of processes to run in parallel")
    args = parser.parse_args()

    #retrieve parsed values
    input_root = args.input_root
    output_root = args.output_root
    gfs_grib2 = args.gfs_grib2
    forcing = join(output_root,args.forcing)
    exactextract_files = join(output_root,args.ExactExtract_csv_files)
    num_processes = args.num_processes
    hyfabfile = args.catchment_source
    hyfab_feature_file = args.catchment_features

    #generate catchment geometry from hydrofabric
    cat_df_full = gpd.read_file(hyfabfile)
    g = [i for i in cat_df_full.geometry]
    h = [i for i in cat_df_full.id]
    n_cats = len(g)
    num_catchments = n_cats
    print("number of catchments = {}".format(n_cats))


    # This file was generated from inital NextGen version 1.2 hydrofabric, but issues are still occuring with GIS buffer methods
    # for hydrofabric team. This is being resolved currently but we have produced a csv file from the HUC01 geopackage forcing 
    # metadata layer that will give us data regarding the NextGen catchment slope, azmiuth angle, elevation, centroid coordinates,
    # and area that will are needed to downscale GFS data from its given grid cells to a given NextGen catchment
    NextGen_catchment_features = pd.read_csv(hyfab_feature_file)
 
    # GFS 12km surface flux data pathway to grib2 files
    datafile_path = join(gfs_grib2, "*.grib2")
    #get list of files
    datafiles = glob.glob(datafile_path)
    print("number of forcing files = {}".format(len(datafiles)))
    #process data with time ordered
    datafiles.sort()

    #prepare for processing
    num_files = len(datafiles)

    # Now we want to create a variable to index the datafile 
    # that are already sorted. This index will assign data
    # to global meterological variables below
    file_index = np.arange(num_files)

    # Put GFS metadata netcdf file in forcing file directory
    # As seperate file to use for netcdf metadata
    gfs_ncfile = join(forcing, "GFS_metadata.nc")


    ####### Right now, the only need and utility for pywgrib 2 #########
    ###### python module is a quick and efficent way to just  #########
    ##### produce a netcdf file and extract GFS metadata to   #########
    ##### configure NextGen netcdf file formatting, but there ########
    ##### is no scale factors and offset within GFS metadata, #######
    ##### so we may just get rid of this in the near future.  #######
 
    # Convert grib2 file that begins after GFS forecast hour 0
    # with precipitation rates initalized to netcdf to extract 
    # metadata from meteorological variables
    err = pywgrib2_s.wgrib2( [datafiles[1], '-match',':(PRATE:surface|DLWRF|DSWRF:surface|PRES:surface|SPFH:2 m above ground|TMP:2 m above ground|UGRD:10 m above ground|VGRD:10 m above ground|HGT:surface):', "-netcdf", gfs_ncfile] )

    # Extract variable names from GFS netcdf data
    nc_file = nc4.Dataset(gfs_ncfile)
    # Get variable list from GFS file
    nc_vars = list(nc_file.variables.keys())
    # Get indices corresponding to Meteorological data
    indices = [nc_vars.index(i) for i in nc_vars if '_' in i]
    # Make array with variable names to use for ExactExtract module
    GFS_met_vars = np.array(nc_vars)[indices]
    
    # get scale_factor and offset keys if available
    add_offset = np.zeros([len(GFS_met_vars)])
    scale_factor = np.zeros([len(GFS_met_vars)])
    i = 0
    for key in GFS_met_vars:
        try:
            scale_factor[i] = nc_file.variables[key].scale_factor
        except AttributeError as e:
            scale_factor[i] = 1.0
        try:
            add_offset[i] = nc_file.variables[key].add_offset
        except AttributeError as e:
            add_offset[i] = 0.0
        i += 1


    # Close netcdf file
    nc_file.close()


    # Get raster transformation (longitude scale to -180 to 180) and subset
    # indices for CONUS data where grid spacing is even and not distorted
    transform, projection, i1_conus, j1_conus, cols_conus, rows_conus = gdal_grib2_transformation(datafiles[2])

    # AORC reference time to use with
    # respect to GFS forecast time
    ref_date = pd.Timestamp("1970-01-01 00:00:00")

    #generate the data objects for child processes
    file_groups = np.array_split(np.array(datafiles), num_processes)
    file_index_groups = np.array_split(file_index, num_processes)

    process_data = []
    process_list = []
    lock = Lock()

    for i in range(num_processes):
        # fill the dictionary with gfs grib2 file and its indices
        data = {}
        data["forcing_files"] = file_groups[i]
        data["file_index"] = file_index_groups[i]
       
        #append to the list
        process_data.append(data)

        p = Process(target=process_sublist, args=(data, lock, i))

        process_list.append(p)

    #start all processes
    for p in process_list:
        p.start()

    #wait for termination
    for p in process_list:
        p.join()

    # Now get file paths for created ExactExtract csv files
    ExactExtract_path = join(exactextract_files, "*.csv")
    weighted_csv_files = glob.glob(ExactExtract_path)
    print("Number of ExactExtract csv files = {}".format(len(weighted_csv_files)))
    #process data with time ordered
    weighted_csv_files.sort()

    # generate single NextGen netcdf file from global variables
    # of regridded GFS data
    create_ngen_netcdf(weighted_csv_files, gfs_ncfile)

