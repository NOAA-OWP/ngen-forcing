#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:15:15 2021

@author: Camaron.George
"""
import os, numpy as np, netCDF4 as nc
import datetime
import pathlib
import pandas as pd
from scipy import spatial
import glob

def extract_inflow_NWM(current_netcdf_filename, NWM_masks):
    with nc.Dataset(current_netcdf_filename) as ncdataa:

        # extract NWM Common ID lateral discharge and return values
        stream_flow_vals = ncdataa['streamflow'][:][NWM_masks]

    return stream_flow_vals

def get_inputfiles(input_path, start, end):
    """Generate list of files between start and end timestamps.

    Args:
        input_path (pathlib.Path): Directory to process
        start (datetime.datetime): Start date
        end (datetime.datetime): End date

    Raises:
        FileNotFoundError: Raised when a missing file is encountered

    Returns:
        (list): List of files matching date pattern
    """
    files = {}
    fdate = start
    one_hour = datetime.timedelta(hours=1)
    # Filter for variable NWM channel route output
    # file extensions
    try:
        chrt = next(input_path.glob("*.CHRTOUT_DOMAIN1"))
        extension = "CHRTOUT_DOMAIN1"
    except:
        chrt = next(input_path.glob("*.CHRTOUT_DOMAIN1.comp"))
        extension = "CHRTOUT_DOMAIN1.comp"
    while fdate <= end:
        cand = chrt.with_name(fdate.strftime('%Y%m%d%H%M.'+extension))
        if cand.exists():
            files[fdate] = cand
        else:
            raise FileNotFoundError(cand)
        fdate += one_hour
    return files

def get_forecast_inputfiles(input_path, start, end, timestamps):
    """Generate list of files between start and end timestamps.

    Args:
        input_path (pathlib.Path): Directory to process
        start (datetime.datetime): Start date
        end (datetime.datetime): End date

    Raises:
        FileNotFoundError: Raised when a missing file is encountered

    Returns:
        (list): List of files matching date pattern
    """

    chrt = glob.glob(str(input_path)+'/*.f*.nc')
    chrt.sort()
    nwm = nc.Dataset(chrt[0])
    reference_time = nwm.variables['time'].units
    nwm.close()

    forecast_timestamps = []
    for i in range(len(chrt)):
        nwm = nc.Dataset(chrt[i])
        forecast_timestamps.append(nwm.variables['time'][:][0])
        nwm.close()

    time_final = pd.to_datetime(nc.num2date(forecast_timestamps,units=reference_time,only_use_cftime_datetimes=False))
    idx = np.in1d(time_final,timestamps)
    chrt_final = np.array(chrt)[idx]

    files = {str(z): chrt_final[z] for z in range(len(chrt_final))}

    return files


# Flag to indicate whether or not using the NWM forecast file
# naming convention with the forecast hour (f01) instead of 
# the NWM timestamp output naming convention
forecast = True


#path to where the source_sink_BMI.in file is stored
filePath = './NWMv3_Atlantic/'

# path to where the NWM output files are stored
nwm_path = '/scratch2/NCEPDEV/ohd/Camaron.George/Data/blueSky/medium_range_mem1/'

nwm_path = pathlib.Path(nwm_path)

# User defined start time and end time for SCHISM simulation
start_time = '2022-08-02 00:00:00'
stop_time = '2022-08-03 00:00:00'


start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
stop_time = datetime.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')


# Calculate the NWM hourly timestamps for a time
# array to be created based on user start and end time
timestamps = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'),end=stop_time.strftime('%Y-%m-%d %H:%M:%S'),freq='h')
time_final = (timestamps - timestamps[0]).total_seconds()


# read in element ids and NWMv3 flow path ids
# for sources and sinks
source_element = []
nwm_source = []
sink_element = []
nwm_sink = []

with open(os.path.join(filePath,'source_sink_BMI.in')) as f:
    nsoel1 = int(f.readline())
    for i in range(nsoel1):
        data = f.readline().split()
        source_element.append(int(data[0]))
        nwm_source.append(str(data[1]))
    next(f)
    nsiel = int(f.readline())
    if(nsiel != 0):
        for i in range(nsiel):
            data = f.readline().split()
            sink_element.append(int(data[0]))
            nwm_sink.append(str(data[1]))

if(forecast):
    # Get all the forecast files based on user start and end
    # time specified above
    files = get_forecast_inputfiles(nwm_path, start_time, stop_time,timestamps)
else:
    # Get all the file names within the NWM CHRTOUT directory and
    # sort them out based on user start and end time
    files = get_inputfiles(nwm_path, start_time, stop_time)

# Now extract only the number based component of the NWM flow path id
source_feature_ids = np.array(nwm_source,dtype=float)
if(nsiel != 0):
    sink_feature_ids = np.array(nwm_sink,dtype=float)


# Find the masks based on NWM sources and sinks
# linked with SCHISM inland boundaries
f0 = next(iter(files.values()))
ncdata = nc.Dataset(f0)
feature_id_index = np.array(ncdata.variables['feature_id'][:].data,dtype=float)
ncdata.close()

feature_ids = np.empty((len(feature_id_index),2))
feature_ids[:,0] = feature_id_index
feature_ids[:,1] = feature_id_index


troute_source_ids = np.empty((len(source_feature_ids),2))
troute_source_ids[:,0] = source_feature_ids
troute_source_ids[:,1] = source_feature_ids

distance, source_mask = spatial.KDTree(feature_ids).query(troute_source_ids)

if(nsiel != 0):
    troute_sink_ids = np.empty((len(sink_feature_ids),2))
    troute_sink_ids[:,0] = sink_feature_ids
    troute_sink_ids[:,1] = sink_feature_ids

    distance, sink_mask = spatial.KDTree(feature_ids).query(troute_sink_ids)


# allocate T-Route streamflow time series arrays
source_inflows = np.ma.masked_array(np.zeros((len(source_feature_ids), len(time_final))), fill_value=0.0)
if(nsiel != 0):
    sink_inflows = np.ma.masked_array(np.zeros((len(sink_feature_ids), len(time_final))), fill_value=0.0)

print("Extracting NWMv3 sources and sinks...")
ffiles = len(files)
for i, f in enumerate(files.values()):
    source_inflows[:,i] = extract_inflow_NWM(f,source_mask)
    if(nsiel != 0):
        sink_inflows[:,i] = extract_inflow_NWM(f,sink_mask)

    print("{}/{} ({:.2%})".format(i, ffiles, i/ffiles).ljust(20), end="\r")


# Flip around the NWM source sink arrays to 
# match expected SCHISM source.nc formatting
so2 = np.zeros((len(time_final),len(source_feature_ids)))
for i in range(len(source_feature_ids)):
    so2[:,i] = source_inflows[i,:]
if(nsiel != 0):
    si = np.zeros((len(time_final),len(sink_feature_ids)))
    for i in range(len(sink_feature_ids)):
        si[:,i] = sink_inflows[i,:]

mso = np.zeros((len(time_final),2,len(source_feature_ids)))
mso[:,0,:] = int(-9999)



if os.path.exists(filePath+'source_NWMv3.nc'):
    os.remove(filePath+'source_NWMv3.nc')
 
#write source.nc file      
ncout = nc.Dataset(filePath+'source_NWMv3.nc','w',format='NETCDF4')

ncout.createDimension('time_vsource',len(time_final))
ncout.createDimension('time_vsink',len(time_final))
ncout.createDimension('time_msource',len(time_final))
ncout.createDimension('nsources',nsoel1)
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

if(nsiel != 0):
    ncsi[:] = sink_element
    ncvsi[:] = si*-1

ncso[:] = source_element
ncvso[:] = so2
ncvmo[:] = mso
nctso[:] = time_final
nctsi[:] = time_final
nctmo[:] = time_final
ncvsos[:] = 3600
ncvsis[:] = 3600
ncvmos[:] = 3600

ncout.close()
