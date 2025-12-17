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

path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Puerto_Rico_Data/'
dataPath = path+'NWM_Forcings/' #path to where the forcing data is stored
sfluxFile = path+'SCHISM/sflux2sourceInput.nc'
out = path+'SCHISM/' #path to where you want the output files stored
#End User Input

#file for rest of atmo variables, which will be used in the simulation so this should be saved in the folder where it will be used in the run
outFile = out+'sflux_air_1.0001.nc'
vFile = out+'source2.nc'

#analysis file/data that will be used for zero time step
files = glob.glob(dataPath+'*')
files.sort()

if 'LDASIN' in files[0]:
    var = 'valid_time'
else:
    var = 'time'

data = nc.Dataset(sfluxFile,'r')
precip2flux = data.variables['precip2flux'][:]
simplex = data.variables['simplex'][:]
area_cor = data.variables['area_cor'][:]
x = data.variables['x'][:]
y = data.variables['y'][:]
        
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
        start = data.variables[var][:]
        ref = data.variables[var].units
        if 'seconds' in ref:
            c1 = 3600.0 #seconds per hour
            c2 = 86400.0 #seconds per day
        elif 'minutes' in ref:
            c1 = 60.0 #minutes per hour
            c2 = 1440.0 #minutes per day
        else:
            print('check reference time')
            exit(1)
        startDate = nc.num2date(start,units=ref,only_use_cftime_datetimes=False)
        baseDateStr = str(startDate[0].year)+'-'+str(startDate[0].month)+'-'+str(startDate[0].day)+' '+str(startDate[0].hour) #the first time in the simulation
        baseDate = [startDate[0].year,startDate[0].month,startDate[0].day,startDate[0].hour] #the first time in the simulation
        ref = 'days since ' + baseDateStr
    t[i,:,:] = data.variables['T2D'][:]
    q[i,:,:] = data.variables['Q2D'][:]
    u[i,:,:] = data.variables['U2D'][:]
    v[i,:,:] = data.variables['V2D'][:]
    p[i,:,:] = data.variables['PSFC'][:]
    r[i,:] = np.sum((data.variables['RAINRATE'][:].flatten()[simplex])*area_cor,axis=1)*precip2flux
    time[i] = (data.variables[var][:]-start)/c2
    if i == len(files)-1:
        # add extra time step to the rest of the variables   
        t[i+1,:,:] = data.variables['T2D'][:]
        q[i+1,:,:] = data.variables['Q2D'][:]
        u[i+1,:,:] = data.variables['U2D'][:]
        v[i+1,:,:] = data.variables['V2D'][:]
        p[i+1,:,:] = data.variables['PSFC'][:]
        time[i+1] = (data.variables[var][:]+c1-start)/c2
 
if os.path.exists(vFile):
    os.remove(vFile)
 
#write source2.nc file      
ncout = nc.Dataset(vFile,'w',format='NETCDF4')

ncout.createDimension('time_vsource',len(time)-1)
ncout.createDimension('nsources',len(precip2flux))

ncvso = ncout.createVariable('vsource','f8',('time_vsource','nsources',))

ncvso[:] = r

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
