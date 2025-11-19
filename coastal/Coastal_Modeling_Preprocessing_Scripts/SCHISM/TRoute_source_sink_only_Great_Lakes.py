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



#path to where the source_sink_BMI.in files are stored for US and Canada
filePath = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Ontario/SCHISM/'

# path to where the T-Route output files are stored
troute_path_US = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Lake_Ontario/TRoute_Lake_Ontario_US'
troute_path_US = pathlib.Path(troute_path_US)
troute_path_Can = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Lake_Ontario/TRoute_Lake_Ontario_Canada'
troute_path_Can = pathlib.Path(troute_path_Can)

# User defined start time and end time for SCHISM simulation
start_time = '2019-04-01 00:00:00'
stop_time = '2019-09-01 00:00:00'


start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
stop_time = datetime.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')

# read in element ids and troute flow path ids
# for sources and sinks
source_element_US = []
troute_source_US = []
sink_element_US = []
troute_sink_US = []

with open(os.path.join(filePath,'source_sink_BMI_US.in')) as f:
    nsoel1_US = int(f.readline())
    for i in range(nsoel1_US):
        data = f.readline().split()
        source_element_US.append(int(data[0]))
        troute_source_US.append(str(data[1]))
    next(f)
    nsiel_US = int(f.readline())
    if(nsiel_US != 0):
        for i in range(nsiel_US):
            data = f.readline().split()
            sink_element_US.append(int(data[0]))
            troute_sink_US.append(str(data[1]))


source_element_Can = []
troute_source_Can = []
sink_element_Can = []
troute_sink_Can = []

with open(os.path.join(filePath,'source_sink_BMI_Can.in')) as f:
    nsoel1_Can = int(f.readline())
    for i in range(nsoel1_Can):
        data = f.readline().split()
        source_element_Can.append(int(data[0]))
        troute_source_Can.append(str(data[1]))
    next(f)
    nsiel_Can = int(f.readline())
    if(nsiel_Can != 0):
        for i in range(nsiel_Can):
            data = f.readline().split()
            sink_element_Can.append(int(data[0]))
            troute_sink_Can.append(str(data[1]))



# Now combine the US (first) and Canada (second) source_sink.in files
# for a finalized source_sink.in file for the given lake domain
with open(os.path.join(filePath,'source_sink_BMI.in'),'w') as out:
    out.write(str(int(nsoel1_US+nsoel1_Can))+"\n")
    for i in range(nsoel1_US):
        out.write(str(source_element_US[i])+ " " + str(troute_source_US[i])+"\n")
    for i in range(nsoel1_Can):
        out.write(str(source_element_Can[i])+ " " + str(troute_source_Can[i])+"\n")
    out.write("\n")
    out.write(str(int(nsiel_US+nsiel_Can))+"\n")
    if(nsiel_US != 0):
        for i in range(nsiel_US):
            out.write(str(sink_element_US[i])+ " " + str(troute_sink_US[i])+"\n")
    if(nsiel_Can != 0):
        for i in range(nsiel_Can):
            out.write(str(sink_element_Can[i])+ " " + str(troute_sink_Can[i])+"\n")

# Get all the file names within the troute repository and
# sort them out based on user start and end time
files_US = get_inputfiles(troute_path_US, start_time, stop_time)
files_Can = get_inputfiles(troute_path_Can, start_time, stop_time)

# Now extract only the number based component of the pois
source_feature_ids_US = np.array(troute_source_US,dtype=float)
if(nsiel_US != 0):
    sink_feature_ids_US = np.array(troute_sink_US,dtype=float)

source_feature_ids_Can = np.array(troute_source_Can,dtype=float)
if(nsiel_Can != 0):
    sink_feature_ids_Can = np.array(troute_sink_Can,dtype=float)


# Find the masks based on T-Route sources and sinks
# linked with SCHISM inland boundaries
f0_US = next(iter(files_US.values()))
ncdata = nc.Dataset(f0_US)
#feature_id_index = ncdata.variables['feature_id'][:]
feature_id_index = ncdata.variables['POI'][:]
feature_id_index = np.where(feature_id_index=='nan',-9999.,feature_id_index)
feature_id_index = np.array(feature_id_index,dtype=float)

feature_ids = np.empty((len(feature_id_index),2))
feature_ids[:,0] = feature_id_index
feature_ids[:,1] = feature_id_index


troute_source_ids = np.empty((len(source_feature_ids_US),2))
troute_source_ids[:,0] = source_feature_ids_US
troute_source_ids[:,1] = source_feature_ids_US

distance, source_mask_US = spatial.KDTree(feature_ids).query(troute_source_ids)

if(nsiel_US != 0):
    troute_sink_ids = np.empty((len(sink_feature_ids_US),2))
    troute_sink_ids[:,0] = sink_feature_ids_US
    troute_sink_ids[:,1] = sink_feature_ids_US

    distance, sink_mask_US = spatial.KDTree(feature_ids).query(troute_sink_ids)

f0_Can = next(iter(files_Can.values()))
ncdata = nc.Dataset(f0_Can)
#feature_id_index = ncdata.variables['feature_id'][:]
feature_id_index = ncdata.variables['POI'][:]
feature_id_index = np.where(feature_id_index=='nan',-9999.,feature_id_index)
feature_id_index = np.array(feature_id_index,dtype=float)

feature_ids = np.empty((len(feature_id_index),2))
feature_ids[:,0] = feature_id_index
feature_ids[:,1] = feature_id_index


troute_source_ids = np.empty((len(source_feature_ids_Can),2))
troute_source_ids[:,0] = source_feature_ids_Can
troute_source_ids[:,1] = source_feature_ids_Can

distance, source_mask_Can = spatial.KDTree(feature_ids).query(troute_source_ids)

if(nsiel_Can != 0):
    troute_sink_ids = np.empty((len(sink_feature_ids_Can),2))
    troute_sink_ids[:,0] = sink_feature_ids_Can
    troute_sink_ids[:,1] = sink_feature_ids_Can

    distance, sink_mask_Can = spatial.KDTree(feature_ids).query(troute_sink_ids)


# Calculate the T-Route 5-minute timestamps for a time
# array to be created based on user start and end time
#timestamps = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'),end=stop_time.strftime('%Y-%m-%d %H:%M:%S'),freq='5min')
timestamps = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'),end=stop_time.strftime('%Y-%m-%d %H:%M:%S'),freq='H')
time_final = (timestamps - timestamps[0]).total_seconds()


# allocate T-Route streamflow time series arrays
source_inflows_timeseries = np.ma.masked_array(np.zeros((int(nsoel1_US+nsoel1_Can), len(files_US)*12)), fill_value=0.0)
source_inflows = np.ma.masked_array(np.zeros((int(nsoel1_US+nsoel1_Can), len(time_final))), fill_value=0.0)
if(int(nsiel_US+nsiel_Can) != 0):
    sink_inflows_timeseries = np.ma.masked_array(np.zeros((int(nsiel_US+nsiel_Can), len(files_US)*12)), fill_value=0.0)
    sink_inflows = np.ma.masked_array(np.zeros((int(nsiel_US+nsiel_Can), len(time_final))), fill_value=0.0)

timestamps_timeseries = []

print("Reading entire T-Route time series...")
ffiles = len(files_US)
files_US_final = np.array(list(files_US.values()),dtype=str)
files_Can_final = np.array(list(files_Can.values()),dtype=str)
troute_index = 0
for i in range(len(files_US_final)):
    f_US = files_US_final[i]
    source_inflows_timeseries[0:nsoel1_US,troute_index:troute_index+12], timestamps_file = extract_inflow_TRoute(f_US,source_mask_US)
    f_Can = files_Can_final[i]
    source_inflows_timeseries[nsoel1_US:,troute_index:troute_index+12], timestamps_file = extract_inflow_TRoute(f_Can,source_mask_Can)
    if(nsiel_US != 0):
        sink_inflows_timeseries[0:nsiel_US,troute_index:troute_index+12], timestamps_file = extract_inflow_TRoute(f_US,sink_mask_US)
        if(nsiel_Can != 0):
            sink_inflows_timeseries[nsiel_US:,troute_index:troute_index+12], timestamps_file = extract_inflow_TRoute(f_Can,sink_mask_Can)
    elif(nsiel_Can != 0):
        sink_inflows_timeseries[0:nsiel_Can,troute_index:troute_index+12], timestamps_file = extract_inflow_TRoute(f_Can,sink_mask_Can)

    troute_index += 12
    timestamps_timeseries.append(timestamps_file)
    #print("{}/{} ({:.2%})".format(i, files_US_final, i/files_US_final).ljust(20), end="\r")

timestamps_timeseries = pd.to_datetime([x for xs in timestamps_timeseries for x in xs])
print("Extracting T-Route sources and sinks...")
for i in range(len(time_final)):
    index = abs((timestamps_timeseries - timestamps[i]).total_seconds()).argmin(0)
    source_inflows[:,i] = source_inflows_timeseries[:,index]
    if(int(nsiel_US+nsiel_Can) != 0):
        sink_inflows[:,i] = sink_inflows_timeseries[:,index]
    print("{}/{} ({:.2%})".format(i, len(time_final), i/len(time_final)).ljust(20), end="\r")



so2 = np.zeros((len(time_final),int(nsoel1_US+nsoel1_Can)))
for i in range(int(nsoel1_US+nsoel1_Can)):
    so2[:,i] = source_inflows[i,:]
if(int(nsiel_US+nsiel_Can) != 0):
    si = np.zeros((len(time_final),int(nsiel_US+nsiel_Can)))
    for i in range(int(nsiel_US+nsiel_Can)):
        si[:,i] = sink_inflows[i,:]

mso = np.zeros((len(time_final),2,int(nsoel1_US+nsoel1_Can)))
mso[:,0,:] = int(-9999)

if os.path.exists(filePath+'source_TRoute.nc'):
    os.remove(filePath+'source_TRoute.nc')
 
#write source.nc file      
ncout = nc.Dataset(filePath+'source_TRoute.nc','w',format='NETCDF4')

ncout.createDimension('time_vsource',len(time_final))
ncout.createDimension('time_vsink',len(time_final))
ncout.createDimension('time_msource',len(time_final))
ncout.createDimension('nsources',int(nsoel1_US+nsoel1_Can))
ncout.createDimension('nsinks',int(nsiel_US+nsiel_Can))
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

if(int(nsiel_US+nsiel_Can) != 0):
    ncsi[:] = sink_element_US + sink_element_Can
    ncvsi[:] = si*-1

ncso[:] = source_element_US + source_element_Can
ncvso[:] = so2
ncvmo[:] = mso
nctso[:] = time_final
nctsi[:] = time_final
nctmo[:] = time_final
ncvsos[:] = 3600
ncvsis[:] = 3600
ncvmos[:] = 3600

ncout.close()
