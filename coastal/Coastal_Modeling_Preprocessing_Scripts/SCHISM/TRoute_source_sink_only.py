#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:15:15 2021

@author: Camaron.George
"""
import os, numpy as np, netCDF4 as nc
import argparse
import datetime
import pathlib
import pandas as pd
from scipy import spatial

def extract_pois(feature_id):
    feature_id_index = np.empty(len(feature_id),dtype=float)
    number = False
    for i in range(len(feature_id_index)):
        if(feature_id[i] != 'nan'):
            if(number == False):
                poi = list(feature_id[i])
                for j in range(len(poi)):
                    if(number != True):
                        try:
                            int(poi[j])
                            domain_slice = ''.join(poi[0:j])
                            number = True
                        except:
                            pass
                feature_id_index[i] = float(feature_id[i].split(domain_slice)[-1])
            else:
                feature_id_index[i] = float(feature_id[i].split(domain_slice)[-1])
        else:
            feature_id_index[i] = np.nan
    return feature_id_index


def binary_isin(elements, test_elements, assume_sorted=False, return_indices=False):
    """Test if values of elements are present in test_elements.
    Returns a boolean array: True if element is present in test_elements, False otherwise.
    If return_indices=True, return an array of indexes that transforms the elements of
    test_elements to same order of elements (for the elements that are present in test_elements).
    ie, for every True in the returned mask, also return the index such that test_elements[indices] == elements[mask]
    """
    elements = np.asarray(elements)
    test_elements = np.asarray(test_elements)
    if assume_sorted:
        idxs = np.searchsorted(test_elements, elements)
    else:
        asorted = test_elements.argsort()
        idxs = np.searchsorted(test_elements, elements, sorter=asorted)
    valid = idxs != len(test_elements)
    test_selector = idxs[valid]
    if not assume_sorted:
        test_selector = asorted[test_selector]
    mask = np.zeros(elements.shape, dtype=bool)
    mask[valid] = test_elements[test_selector] == elements[valid]
    #indices are the array of indexes that transform the elements in
    # test_elements to match the order of elements
    if return_indices:
        return mask, test_selector[mask[valid]]
    else:
        return mask

# this function extracts the offsest values for the requested ids
def extract_offsets(current_netcdf_filename, selected_ids):
    """Return a mask on feature_id for the selected_ids
    Args:
        current_netcdf_filename (str): NetCDF file to read
        selected_ids (list_like): ids to select.

    Returns:
        (ndarray): Indexes *in order* of selected comm_ids.
    """
    with nc.Dataset(current_netcdf_filename) as ncdata:
        #feature_id_index = ncdata['POI'][:]
        #feature_id_index = np.where(feature_id_index=='nan',np.nan,feature_id_index)
        #feature_id_index = np.array(feature_id_index,dtype=float)

        feature_id_index = ncdata['feature_id'][:]
        #print("Size of feature id data is ", len(feature_id_index))
        fmask, fidx = binary_isin(feature_id_index, selected_ids, return_indices=True)
        fvals = feature_id_index[fmask]
        return fmask, fvals, fidx

def extract_inflow_TRoute(current_netcdf_filename,masks):
    with nc.Dataset(current_netcdf_filename) as ncdata:
        # extract NWM Common ID lateral discharge and return values
        stream_flow_vals = ncdata['flow'][:].data[masks,:]
        timestamps = nc.num2date(ncdata['time'][:],units='seconds since '+ncdata.file_reference_time,only_use_cftime_datetimes=False)
        #timestamps = nc.num2date(ncdata['time'][:],units='seconds since 2024-02-19_00:00:00',only_use_cftime_datetimes=False)

    return stream_flow_vals, timestamps.data

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
    chrt = next(input_path.glob("troute_output_*"))
    troute_name = "troute_output_"
    extension = ".nc"
    while fdate <= end:
        cand = chrt.with_name(fdate.strftime(troute_name + '%Y%m%d%H%M'+extension))
        if cand.exists():
            files[fdate] = cand
        else:
            raise FileNotFoundError(cand)
        fdate += one_hour
    return files


parser = argparse.ArgumentParser()
parser.add_argument('filePath', type=str, help='path to the source_sink_BMI.in file')
parser.add_argument('troutepath', type=str, help='t-route output path')
parser.add_argument('start_time', type=str, help='Start time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 00:00"')
parser.add_argument('end_time', type=str, help='End time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 23:00"')
parser.add_argument('domain', type=str, help='domain name, one of hawaii, prvi, pacfic, or atlgulf')
args = parser.parse_args()

#path to where the source_sink_BMI.in file is stored
#filePath = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Alaska/SCHISM/'
#filePath = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Hawaii/SCHISM/POIs_No_Anamolies/'
#filePath = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/PR+USVI/SCHISM/'
#filePath = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Atlantic+Gulf/SCHISM/'
filePath = args.filePath
#filePath = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Pacific/SCHISM/'

# path to where the T-Route output files are stored
#troute_path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Alaska_DFLOWFM/TRoute'
#troute_path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Hawaii_Data/TRoute'
#troute_path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Puerto_Rico_Data/TRoute'
#troute_path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Atlantic_Data/TRoute'
troute_path = args.troutepath
#troute_path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Pacific_Data/TRoute'

troute_path = pathlib.Path(troute_path)

# User defined start time and end time for SCHISM simulation
#start_time = '2002-11-01 00:00:00'
#stop_time = '2002-11-14 23:00:00'
#start_time = '2003-08-28 00:00:00'
#stop_time = '2003-09-04 23:00:00'
#start_time = '2017-09-01 00:00:00'
#stop_time = '2017-10-14 23:00:00'
#start_time = '2018-09-12 00:00:00'
#stop_time = '2018-09-22 00:00:00'
#start_time = '2014-07-31 00:00:00'
#stop_time = '2014-08-09 00:00:00'
start_time = args.start_time
stop_time = args.end_time

start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
stop_time = datetime.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')

# read in element ids and troute flow path ids
# for sources and sinks
source_element = []
troute_source = []
sink_element = []
troute_sink = []

with open(os.path.join(filePath,'source_sink_BMI.in')) as f:
    nsoel1 = int(f.readline())
    for i in range(nsoel1):
        data = f.readline().split()
        source_element.append(int(data[0]))
        troute_source.append(str(data[1]))
    next(f)
    nsiel = int(f.readline())
    if(nsiel != 0):
        for i in range(nsiel):
            data = f.readline().split()
            sink_element.append(int(data[0]))
            troute_sink.append(str(data[1]))

# Get all the file names within the troute repository and
# sort them out based on user start and end time
files = get_inputfiles(troute_path, start_time, stop_time)

# Now extract only the number based component of the pois
#source_feature_ids = extract_pois(troute_source)
source_feature_ids = np.array(troute_source,dtype=float)
if(nsiel != 0):
    #sink_feature_ids = extract_pois(troute_sink)
    sink_feature_ids = np.array(troute_sink,dtype=float)


#inflow_boundaries = pd.read_csv('/scratch2/NCEPDEV/ohd/Jason.Ducker/Hawaii_Data/HI_InflowsOutflows_hfv2_noAnomalies.csv')
#source_feature_ids = []
#for i in range(len(troute_source)):
#    idx = np.where(inflow_boundaries.hl_link.values == troute_source[i])[0][0]
#    source_feature_ids.append(float(inflow_boundaries.flowpath_id.values[idx].split('-')[-1]))

#if(nsiel != 0):
#    sink_feature_ids = []
#    for i in range(len(troute_sink)):
#        idx = np.where(inflow_boundaries.hl_link.values == troute_sink[i])[0][0]
#        sink_feature_ids.append(float(inflow_boundaries.flowpath_id.values[idx].split('-')[-1]))


# Find the masks based on T-Route sources and sinks
# linked with SCHISM inland boundaries
f0 = next(iter(files.values()))
ncdata = nc.Dataset(f0)
feature_id_index = ncdata.variables['feature_id'][:]
#feature_id_index = ncdata.variables['POI'][:]
feature_id_index = np.where(feature_id_index=='nan',-9999.,feature_id_index)
feature_id_index = np.array(feature_id_index,dtype=float)

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


#source_mask, source_feature_ids, fidx = extract_offsets(f0,troute_source_ids)
#if(nsiel != 0):
#    sink_mask, sink_feature_ids, fidx = extract_offsets(f0,troute_sink_ids)



# Calculate the T-Route 5-minute timestamps for a time
# array to be created based on user start and end time
#timestamps = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'),end=stop_time.strftime('%Y-%m-%d %H:%M:%S'),freq='5min')
timestamps = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'),end=stop_time.strftime('%Y-%m-%d %H:%M:%S'),freq='H')
time_final = (timestamps - timestamps[0]).total_seconds()


# allocate T-Route streamflow time series arrays
source_inflows_timeseries = np.ma.masked_array(np.zeros((len(source_feature_ids), len(files)*12)), fill_value=0.0)
source_inflows = np.ma.masked_array(np.zeros((len(source_feature_ids), len(time_final))), fill_value=0.0)
if(nsiel != 0):
    sink_inflows_timeseries = np.ma.masked_array(np.zeros((len(sink_feature_ids), len(files)*12)), fill_value=0.0)
    sink_inflows = np.ma.masked_array(np.zeros((len(sink_feature_ids), len(time_final))), fill_value=0.0)

timestamps_timeseries = []

print("Reading entire T-Route time series...")
ffiles = len(files)
troute_index = 0
for i, f in enumerate(files.values()):
    source_inflows_timeseries[:,troute_index:troute_index+12], timestamps_file = extract_inflow_TRoute(f,source_mask)
    if(nsiel != 0):
        sink_inflows_timeseries[:,troute_index:troute_index+12], timestamps_file = extract_inflow_TRoute(f,sink_mask)
    troute_index += 12
    timestamps_timeseries.append(timestamps_file)
    print("{}/{} ({:.2%})".format(i, ffiles, i/ffiles).ljust(20), end="\r")

timestamps_timeseries = pd.to_datetime([x for xs in timestamps_timeseries for x in xs])
print("Extracting T-Route sources and sinks...")
for i in range(len(time_final)):
    index = abs((timestamps_timeseries - timestamps[i]).total_seconds()).argmin(0)
    source_inflows[:,i] = source_inflows_timeseries[:,index]
    if(nsiel != 0):
        sink_inflows[:,i] = sink_inflows_timeseries[:,index]
    print("{}/{} ({:.2%})".format(i, len(time_final), i/len(time_final)).ljust(20), end="\r")


so2 = np.zeros((len(time_final),len(source_feature_ids)))
for i in range(len(source_feature_ids)):
    so2[:,i] = source_inflows[i,:]
if(nsiel != 0):
    si = np.zeros((len(time_final),len(sink_feature_ids)))
    for i in range(len(sink_feature_ids)):
        si[:,i] = sink_inflows[i,:]

mso = np.zeros((len(time_final),2,len(source_feature_ids)))
mso[:,0,:] = int(-9999)

if os.path.exists(filePath+'source_TRoute.nc'):
    os.remove(filePath+'source_TRoute.nc')

print( "write source.nc filewrite source.nc file" )
#write source.nc file      
ncout = nc.Dataset(filePath+'source_TRoute.nc','w',format='NETCDF4')
print( filePath+'source_TRoute.nc')

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
