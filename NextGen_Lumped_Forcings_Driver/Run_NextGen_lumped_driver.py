############# Sample Python script to execute the NextGen lumped forcings driver for NextGen forcing file production ######################
from NextGen_lumped_forcings_driver import NextGen_lumped_forcings_driver
import time

start_time = time.time()

###################### Python executable sample code for AORC Forcings for hydrofabric subset #############################
NextGen_lumped_forcings_driver("./output",start_time="2023-01-05 00:00:00", end_time="2023-01-06 00:00:00", met_dataset="AORC",hyfabfile="./nextgen_01.gpkg", met_dataset_pathway="./AORC_Forcings/",weights_file=None,netcdf=True,csv=True,bias_calibration=False,downscaling=False,CONUS=False,AnA=False,num_processes=5)

###################### Python executable sample code for AORC Forcings for CONUS hydrofabric #############################
NextGen_lumped_forcings_driver("./output",start_time="2023-01-05 00:00:00", end_time="2023-01-06 00:00:00", met_dataset="AORC",hyfabfile="./conus.gpkg", met_dataset_pathway="./AORC_Forcings/",weights_file=None,netcdf=True,csv=False,bias_calibration=False,downscaling=False,CONUS=True,AnA=False,num_processes=5)

###################### Python executable sample code for AORC Forcings for hydrofabric subset on National Water Center (NWC) servers with no AORC forcing dataset required #############################
NextGen_lumped_forcings_driver("./output",start_time="2023-01-05 00:00:00", end_time="2023-01-06 00:00:00", met_dataset="AORC",hyfabfile="./nextgen_01.gpkg", met_dataset_pathway=None,netcdf=True,csv=True,bias_calibration=False,downscaling=False,CONUS=False,AnA=False,num_processes=5)


###################### Python executable sample code for GFS Forcings Medium range for any hydrofabric geopackage #############################
NextGen_lumped_forcings_driver("./output",start_time=None, end_time=None, met_dataset="GFS",hyfabfile="./nextgen_01.gpkg", met_dataset_pathway="./GFS_Forecast_00z_Cycle/",weights_file=None,netcdf=True,csv=True,bias_calibration=False,downscaling=False,CONUS=False,AnA=False,num_processes=5)


###################### Python executable sample code for CFS Forcings Long range for any hydrofabric geopackage #############################
NextGen_lumped_forcings_driver("./output",start_time=None, end_time=None, met_dataset="CFS",hyfabfile="./nextgen_01.gpkg", met_dataset_pathway="./CFS_Forecast_00z_Cycle/",weights_file=None,netcdf=True,csv=True,bias_calibration=False,downscaling=False,CONUS=False,AnA=False,num_processes=5)


###################### Python executable sample code for HRRR Forcings Short range configuration for any hydrofabric geopackage #############################
NextGen_lumped_forcings_driver("./output",start_time=None, end_time=None, met_dataset="HRRR",hyfabfile="./nextgen_01.gpkg", met_dataset_pathway="./HRRR_Data_Directory/",weights_file=None,netcdf=True,csv=True,bias_calibration=False,downscaling=False,CONUS=False,AnA=False,num_processes=5)
# Note, The met_data_pathway="./HRRR_Data_Directory/" data pathway here is just for the pathway to where all the HRRR daily forecast cycles are located (e.g. hrrr.20230104  hrrr.20230105, etc. directoires that are located inside the HRRR_Data_Directory pathway)

###################### Python executable sample code for HRRR Forcings Analysis and Assimilation (AnA) configuration for any hydrofabric geopackage #############################
NextGen_lumped_forcings_driver("./output",start_time=None, end_time=None, met_dataset="HRRR",hyfabfile="./nextgen_01.gpkg", met_dataset_pathway="./HRRR_Data_Directory/",weights_file=None,netcdf=True,csv=True,bias_calibration=False,downscaling=False,CONUS=False,AnA=True,num_processes=5)
# Note, The met_data_pathway="./HRRR_Data_Directory/" data pathway here is just for the pathway to where all the HRRR daily forecast cycles are located (e.g. hrrr.20230104  hrrr.20230105, etc. directoires that are located inside the HRRR_Data_Directory pathway)


end_time = time.time()

print("Seconds it took to complete NextGen Lumped Forcings Driver")
print(end_time - start_time)

###########################################################################################################################################
