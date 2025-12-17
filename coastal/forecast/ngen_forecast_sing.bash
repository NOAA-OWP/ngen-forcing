#!/usr/bin/env bash
set -x
#
#Start time 
export STARTPDY=20250513
#
#Start cycle 
export STARTCYC=05
#
#Forecast length in hours
export LENGTH_HRS=12

#The working directory
export SCHISM_FORECAST_DIR=/efs/$LOGNAME/schism_forecast_test

#The Singularity container
export SING_SIF_PATH=/efs/$LOGNAME/ngen-app/singularity/ngen_coastal_sing.sif

repo=/ngen-app/ngen-forcing

export BINDINGS="/efs"
 
#
#Create the MSMF mesh for the domain
#
singularity exec -B $BINDINGS $SING_SIF_PATH /bin/bash -c \
	"conda run -n ngen_esmf_mesh_domain --no-capture-output \
        python ${repo}/ESMF_Mesh_Domain_Configuration_Production/NextGen_hyfab_to_ESMF_Mesh.py \
     $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/domain/LowerColorado_v22_no_lakes.gpkg \
     $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/domain/LowerColorado_v22_no_lakes_mesh.nc"

export START_TIME="${STARTPDY:0:4}-${STARTPDY:4:2}-${STARTPDY:6} ${STARTCYC}:00:00"
start_timestamp=$(date -u -d "${STARTPDY} ${STARTCYC}" +"%s")
itime=$(( 10#${LENGTH_HRS} * 3600 + $start_timestamp ))
export END_TIME="$(date -u -d "@${itime}" +"%Y-%m-%d %H:00:00")"

#
#Update the forcing configuration file
#
sed -i -e 's|InputForcingDirectories: .*|InputForcingDirectories: \[\"'"$SCHISM_FORECAST_DIR"'\/hrrr\"\]|' \
       -e 's|ScratchDir: .*|ScratchDir: \"'"$SCHISM_FORECAST_DIR"'\/hrrr_scratch\"|' \
	$SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/sr_config.yml

#
#Extract forcings for ngen run
#
singularity exec -B $BINDINGS \
	  --pwd /ngen-app  $SING_SIF_PATH \
	 conda run -n ngen_forcings_engine_bmi --no-capture-output \
    python "${repo}/NextGen_Forcings_Engine_BMI/run_bmi_model.py" \
    -config_path $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/sr_config.yml \
    -geogrid $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/domain/LowerColorado_v22_no_lakes_mesh.nc \
    -b_date ${STARTPDY}${STARTCYC}00 \
    -output_path $SCHISM_FORECAST_DIR/hrrr_dump.nc \
     "$START_TIME" \
     "$END_TIME"

#
#Convert NetCDF forcing file to CSV files
#
singularity exec -B $BINDINGS \
	  --pwd /ngen-app  \
        singularity/ngen_coastal_sing.sif \
conda run -n ngen_forcings_engine_bmi --no-capture-output \
    python "${repo}/NextGen_Forcings_Engine_BMI/post_process/netcdf_to_csv.py" \
     $SCHISM_FORECAST_DIR/hrrr_dump.nc \
     $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/forcings

#
#Now we need to update the ngen realization file and t-route config
# file to use the correct date and time
#
itime=$(( 3600 + $start_timestamp ))
start_time="$(date -u -d "@${itime}" +"%Y-%m-%d %H:00:00")"
itime=$(( 10#${LENGTH_HRS} * 3600 + 3600 + $start_timestamp ))
end_time="$(date -u -d "@${itime}" +"%Y-%m-%d %H:00:00")"

#set start time for the realization
sed -i -e \
     's/^\s*\"start_time\": .*/     \"start_time\": \"'"$start_time"'\",/' \
     $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/Lower_Colorado_River_ngen.json

#set end time for the realization
sed -i -e \
    's/^\s*\"end_time\": .*/     \"end_time\": \"'"$end_time"'\",/' \
    $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/Lower_Colorado_River_ngen.json

#set start time for the t-route configuration
sed -i -e \
    's/^        start_datetime              : .*/        start_datetime              : \"'"$start_time"'\"/' \
    $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/test_AnA_V4_HYFeature_v22.yaml

nts=$(($LENGTH_HRS * 12))

#set nts for the t-route configuration
sed -i -e \
    's/^        nts                         : .*/        nts                         : '"$nts"' # 288 for 1day; 144 for 12 hours/' \
    $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/test_AnA_V4_HYFeature_v22.yaml

#set max_loop_size for the t-route configuration
sed -i -e \
    's/^        max_loop_size               : .*/        max_loop_size               : '"$LENGTH_HRS"' # \[hr\]/' \
    $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen/test_AnA_V4_HYFeature_v22.yaml

mkdir -p /efs/$LOGNAME/ngencerf; \

export BINDINGS="/efs,/efs/$LOGNAME/ngencerf:/ngencerf"

#
#Run ngen simulation
#
singularity exec -B $BINDINGS \
	  --pwd $SCHISM_FORECAST_DIR/Lower_Colorado_River_ngen  \
        singularity/ngen_coastal_sing.sif \
	/bin/bash -c "source /ngen-app/ngen-python/bin/activate; \
	export LD_LIBRARY_PATH=/ngen-app/ngen-python/lib:\$LD_LIBRARY_PATH; \
        /ngen-app/ngen/cmake_build/ngen ./domain/LowerColorado_v22_no_lakes.gpkg \"all\" \
	./domain/LowerColorado_v22_no_lakes.gpkg \"all\" ./Lower_Colorado_River_ngen.json; deactivate"

