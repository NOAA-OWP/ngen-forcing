#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:54:57 2022

@author: Camaron.George
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:43:00 2021

@author: Camaron.George
"""
import os, glob, numpy as np, netCDF4 as nc
import pandas as pd
import datetime
c1 = 1 #min/hr, sec/hr, or hr depending on the time variable in the wind files
c2 = 86400 #min/day, sec/day, or hr/day depending on the time variable in the wind files

path = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Ontario/SCHISM/'
dataPath = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Lake_Ontario/AORC/'
#path to where the forcing data is stored
sfluxFile = path+'sflux2sourceInput.nc'
out = path #path to where you want the output files stored

start_time = "2019-04-01 00:00:00"
end_time = "2019-09-01 00:00:00"



start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
AORC_datetimes = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'),end=end_time.strftime('%Y-%m-%d %H:%M:%S'),freq='h')

#End User Input

#file for rest of atmo variables, which will be used in the simulation so this should be saved in the folder where it will be used in the run
outFile = out+'sflux_air_1.0001.nc'
vFile = out+'source2.nc'

#analysis fil))/data that will be used for zero time step
files = glob.glob(dataPath+'*.nc*')
files.sort()

AORC_indices = []
for i in range(len(AORC_datetimes)):
    print(AORC_datetimes[i].strftime('%Y%m%d%H'))
    AORC_indices.append([idx for idx, s in enumerate(files) if AORC_datetimes[i].strftime('%Y%m%d%H') in s][0])

files = np.array(files)[AORC_indices]


data = nc.Dataset(sfluxFile,'r')
precip2flux = data.variables['precip2flux'][:]
simplex = data.variables['simplex'][:]
area_cor = data.variables['area_cor'][:]
x = data.variables['x'][:]
y = data.variables['y'][:]
I = data.variables['minMax'][:]
        
t = np.zeros((len(files)+1,x.shape[0],x.shape[1]))
q = np.zeros((len(files)+1,x.shape[0],x.shape[1]))
u = np.zeros((len(files)+1,x.shape[0],x.shape[1]))
v = np.zeros((len(files)+1,x.shape[0],x.shape[1]))
p = np.zeros((len(files)+1,x.shape[0],x.shape[1]))
r = np.zeros((len(files),len(precip2flux)))
time = np.zeros((len(files)+1))

for i in range(len(files)):
    data = nc.Dataset(files[i],'r')
    if i == 0:
        start = data.variables['time'][:]
        ref = data.variables['time'].units
        startDate = nc.num2date(start,units=ref,only_use_cftime_datetimes=False)
        baseDateStr = str(startDate[0].year)+'-'+str(startDate[0].month)+'-'+str(startDate[0].day)+' '+str(startDate[0].hour) #the first time in the simulation
        baseDate = [startDate[0].year,startDate[0].month,startDate[0].day,startDate[0].hour] #the first time in the simulation
        ref = 'days since ' + baseDateStr
        
    t[i,:,:] = data.variables['TMP_2maboveground'][:,I[0]:I[1],I[2]:I[3]]
    q[i,:,:] = data.variables['SPFH_2maboveground'][:,I[0]:I[1],I[2]:I[3]]
    u[i,:,:] = data.variables['UGRD_10maboveground'][:,I[0]:I[1],I[2]:I[3]]
    v[i,:,:] = data.variables['VGRD_10maboveground'][:,I[0]:I[1],I[2]:I[3]]
    p[i,:,:] = data.variables['PRES_surface'][:,I[0]:I[1],I[2]:I[3]]
    r[i,:] = np.sum((data.variables['APCP_surface'][:,I[0]:I[1],I[2]:I[3]].flatten()[simplex])*area_cor,axis=1)*precip2flux
    time[i] = (data.variables['time'][:]-start)/c2
    if i == len(files)-1:
        # add extra time step to the rest of the variables   
        t[i+1,:,:] = data.variables['TMP_2maboveground'][:,I[0]:I[1],I[2]:I[3]]
        q[i+1,:,:] = data.variables['SPFH_2maboveground'][:,I[0]:I[1],I[2]:I[3]]
        u[i+1,:,:] = data.variables['UGRD_10maboveground'][:,I[0]:I[1],I[2]:I[3]]
        v[i+1,:,:] = data.variables['VGRD_10maboveground'][:,I[0]:I[1],I[2]:I[3]]
        p[i+1,:,:] = data.variables['PRES_surface'][:,I[0]:I[1],I[2]:I[3]]
        time[i+1] = (data.variables['time'][:]+c1-start)/c2
 
if os.path.exists(vFile):
    os.remove(vFile)
 
#write source2.nc file      
ncout = nc.Dataset(vFile,'w',format='NETCDF4')

ncout.createDimension('time_vsource',len(time)-1)
ncout.createDimension('nsources',len(precip2flux))

ncvso = ncout.createVariable('vsource','f8',('time_vsource','nsources',))

ncvso[:] = r/3600.0

ncout.close()

if os.path.exists(outFile):
    os.remove(outFile)
            
#write atmo input file
ncout = nc.Dataset(outFile,'w',format='NETCDF4')

ncout.createDimension('time',len(time))
ncout.createDimension('ny_grid',x.shape[0])
ncout.createDimension('nx_grid',x.shape[1])

nctime = ncout.createVariable('time','f4',('time',))
nclon = ncout.createVariable('lon','f4',('ny_grid','nx_grid',))
nclat = ncout.createVariable('lat','f4',('ny_grid','nx_grid',))
ncu = ncout.createVariable('uwind','f4',('time','ny_grid','nx_grid',))
ncv = ncout.createVariable('vwind','f4',('time','ny_grid','nx_grid',))
ncp = ncout.createVariable('prmsl','f4',('time','ny_grid','nx_grid',))
nct = ncout.createVariable('stmp','f4',('time','ny_grid','nx_grid',))
ncq = ncout.createVariable('spfh','f4',('time','ny_grid','nx_grid',))

nctime[:] = time
nctime.long_name = "Time"
nctime.standard_name = "time"
nctime.units = ref
nctime.base_date = baseDate
nclon[:] = x
nclon.long_name = "Longitude"
nclon.standard_name = "longitude"
nclon.units = "degrees_east"
nclat[:] = y
nclat.long_name = "Latitude"
nclat.standard_name = "latitude"
nclat.units = "degrees_north"
ncu[:] = u
ncu.long_name = "Surface Eastward Air Velocity (10m AGL)"
ncu.standard_name = "eastward_wind"
ncu.units = "m/s"
ncv[:] = v
ncv.long_name = "Surface Northward Air Velocity (10m AGL)"
ncv.standard_name = "northward_wind"
ncv.units = "m/s"
ncp[:] = p
ncp.long_name = "Pressure reduced to MSL"
ncp.standard_name = "air_pressure_at_sea_level"
ncp.units = "Pa"
nct[:] = t
nct.long_name = "Surface Air Temperature (2m AGL)"
nct.standard_name = "air_temperature"
nct.units = "K"
ncq[:] = q
ncq.long_name = "Surface Specific Humidity (2m AGL)"
ncq.standard_name = "specific_humidity"
ncq.units = "kg/kg"

ncout.close()
