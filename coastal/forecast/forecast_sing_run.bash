#!/usr/bin/env bash

#SBATCH --job-name=sing_mpi  #job name
#SBATCH -N 4                     #number of nodes to use
#SBATCH --partition=compute      #the patition
#SBATCH --ntasks-per-node=18     #numebr of cores per node
#SBATCH --exclusive 

set -x

export NODES=4         #this must match the number of nodes defined above by slurm
export NCORES=18        #this must match the number of cores per node defined above by slurm
export NPROCS=$((NODES*NCORES))

. ./schism_forecast.cfg

export NGWPC_COASTAL_PARM_DIR=/efs/ngwpc-coastal

export NGEN_APP_DIR=/ngen-app
#
# define the model time step in seconds
export FCST_TIMESTEP_LENGTH_SECS=3600
#

export USHnwm=$NGEN_APP_DIR/nwm.v3.0.6/ush
export PARMnwm=$NGWPC_COASTAL_PARM_DIR/parm
export EXECnwm=$NGEN_APP_DIR/nwm.v3.0.6/exec
export DATAexec=$COASTAL_WORK_DIR

export SAVE_ALL_TASKS=yes
export OMP_NUM_THREADS=2
export IOBUF_PARAMS='*.LAKEOUT_DOMAIN1:size=64M:count=2:prefetch=1,*:size=32M:count=4:vbuffer_count=4096:prefetch=1'
#export IOBUF_PARAMS='*:verbose'
export OMP_PLACES=cores
# ----------------: added (WCOSS2/Pete)
# set up MPI connections and buffers at start of run - helps efficiency of MPI later in run
export MPICH_OFI_STARTUP_CONNECT=1
# pace MPI_Bcast messaging when reading and distributing initial conditions - prevents the Bcast hangs
export MPICH_COLL_SYNC=MPI_Bcast
# turn off MPI_Reduce on node optimization - prevent MPI_Reduce hangs during time stepping
export MPICH_REDUCE_NO_SMP=1

export FI_OFI_RXM_SAR_LIMIT=3145728
export FI_MR_CACHE_MAX_COUNT=0
export FI_EFA_RECVWIN_SIZE=65536

# User specific aliases and functions
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
#

export NFS_MOUNT=/efs
export PATH=/opt/conda/bin:${PATH}
export CONDA_ENVS_PATH=$NFS_MOUNT/ngen-app/conda/envs
export CONDA_ENV_NAME=ngen_forcing_coastal
export COASTAL_VENV=${NGEN_APP_DIR}/ngen-forcing-coastal
export PATH=${CONDA_ENVS_PATH}/${CONDA_ENV_NAME}/bin:${PATH}

conda activate ${CONDA_ENVS_PATH}/$CONDA_ENV_NAME

export PATH=/contrib/software/python_3_10_14/bin:${PATH}
export LD_LIBRARY_PATH=/contrib/software/python_3_10_14/lib:/contrib/software/netcdf/4.7.4/lib:/contrib/software/hdf5/1.12.3/lib:/opt/conda/lib:${CONDA_ENVS_PATH}/lib:$LD_LIBRARY_PATH

export MPICOMMAND2="mpiexec -n ${NPROCS} "
export MPICOMMAND3="mpiexec -n 4 "

declare -A coastal_domain_to_inland_domain=( \
	   [prvi]="domain_puertorico" \
	   [hawaii]="domain_hawaii" \
	   [atlgulf]="domain" \
	   [pacific]="domain" )

declare -A coastal_domain_to_geo_grid=( \
	   [prvi]="geo_em_PRVI.nc" \
	   [hawaii]="geo_em_HI.nc" \
	   [atlgulf]="geo_em_CONUS.nc" \
	   [pacific]="geo_em_CONUS.nc" )

export SCHISM_ESMFMESH=${PARMnwm}/coastal/${COASTAL_DOMAIN}/hgrid.nc
export GEOGRID_FILE=${PARMnwm}/${coastal_domain_to_inland_domain[$COASTAL_DOMAIN]}/${coastal_domain_to_geo_grid[$COASTAL_DOMAIN]}

export DATAlogs=$DATAexec

if [[ ! -d $DATAexec ]]; then
   mkdir -p $DATAexec
fi

export NSCRIBES=2

export BINDINGS="/contrib,$NFS_MOUNT,$CONDA_ENVS_PATH,$NGWPC_COASTAL_PARM_DIR,/usr/bin/bc,/usr/bin/srun,/usr/lib64/libpmi2.so,/usr/lib64/libefa.so,/usr/lib64/libibmad.so,/usr/lib64/libibnetdisc.so,/usr/lib64/libibumad.so,/usr/lib64/libibverbs.so,/usr/lib64/libmana.so,/usr/lib64/libmlx4.so,/usr/lib64/libmlx5.so,/usr/lib64/librdmacm.so"

work_dir=${NGEN_APP_DIR}/ngen-forcing/coastal/forecast
calib_work_dir=${NGEN_APP_DIR}/ngen-forcing/coastal/calib
export COASTAL_PREPROCESSING_SCRIPT_DIR=$work_dir/../

export LENGTH_HRS=$FCST_LENGTH_HRS
export FORCING_BEGIN_DATE=${STARTPDY}${STARTCYC}00
export START_TIME="${STARTPDY:0:4}-${STARTPDY:4:2}-${STARTPDY:6} ${STARTCYC}:00:00"

start_timestamp=$(date -u -d "${STARTPDY} ${STARTCYC}" +"%s")
itime=$(( 10#${LENGTH_HRS} * 3600 + $start_timestamp ))
export FORCING_END_DATE=$(date -u -d "@${itime}" +"%Y%m%d%H00")
export END_TIME="$(date -u -d "@${itime}" +"%Y-%m-%d %H:00:00")"

singularity exec -B $BINDINGS --pwd ${work_dir} $SIF_PATH \
	 ./run_forecast_nexgen_preprocessing.bash

export CYCLE_DATE=$STARTPDY
export CYCLE_TIME=${STARTCYC}00
export LENGTH_HRS=$(singularity exec -B $BINDINGS \
	  --pwd ${work_dir} \
         $SIF_PATH \
	 ./run_sing_coastal_workflow_pre_make_stofs_ocean.bash)

export ESTOFS_INPUT_FILE=$STOFS_FILE
export SCHISM_OUTPUT_FILE=$DATAexec/elev2D.th.nc
export OPEN_BNDS_HGRID_FILE=$DATAexec/open_bnds_hgrid.nc

${MPICOMMAND2} singularity exec -B $BINDINGS \
	  --pwd ${work_dir} \
         $SIF_PATH \
	 $CONDA_ENVS_PATH/$CONDA_ENV_NAME/bin/python \
         $USHnwm/wrf_hydro_workflow_dev/coastal/regrid_estofs.py $ESTOFS_INPUT_FILE $OPEN_BNDS_HGRID_FILE $SCHISM_OUTPUT_FILE 

singularity exec -B $BINDINGS \
	  --pwd ${work_dir} \
         $SIF_PATH \
	 ./run_sing_coastal_workflow_post_make_stofs_ocean.bash

singularity exec -B $BINDINGS \
	  --pwd ${work_dir} \
         $SIF_PATH \
	 ./run_sing_coastal_workflow_pre_schism.bash

export PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin

export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/opt/amazon/openmpi/lib64
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

${MPICOMMAND2} singularity exec -B $BINDINGS --pwd $COASTAL_WORK_DIR \
         $SIF_PATH \
	/bin/bash -c "/ngen-app/nwm.v3.0.6/exec/pschism_wcoss2_NO_PARMETIS_TVD-VL.openmpi $NSCRIBES"

singularity exec -B $BINDINGS \
	  --pwd ${calib_work_dir} \
         $SIF_PATH \
	 ./run_sing_coastal_workflow_post_schism.bash
