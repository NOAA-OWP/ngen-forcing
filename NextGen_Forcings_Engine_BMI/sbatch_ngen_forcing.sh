#!/bin/bash

#SBATCH -n 5
#SBATCH --time=00:30:00

#SBATCH --job-name=ngen-forcing

#SBATCH --account=ohd
#SBATCH --error=ngen_forcing_error.log
#SBATCH --output=ngen_forcing_output.log

#SBATCH --exclusive

module load intel-oneapi-compilers/2025.2.1
module load intel-oneapi-mpi/2021.16.1
module load netcdf-fortran
module load wgrib2

export FC=mpiifx
export CC=mpiicx
export CXX=mpiicpx

export ESMFMKFILE=/scratch3/NCEPDEV/ohd/Jason.Ducker/esmf/lib/libO/Linux.intel.64.intelmpi.default/esmf.mk
export WGRIB2=/apps/wgrib2/3.1.3/gnu_11.4.1/wmo/bin/wgrib2

export PYTHONPATH=$PYTHONPATH:$(pwd)/nextgen_forcings_ewts/src

export PYTHON=/scratch4/NCEPDEV/ohd/Jason.Ducker/miniconda3/envs/ngen_engine/bin/python

export TOTAL_CORES=$(($SLURM_JOB_NUM_NODES * $SLURM_CPUS_ON_NODE))

echo -n "Begin time: "
date

#------------------------------------------------------------
# Run the program

#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/standard_ana_config.yml -output_path ./Scratch/AnA -np $SLURM_NTASKS
#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/standard_ana_config_subhourly.yml -output_path ./Scratch/AnA_SubHourly -np $SLURM_NTASKS
#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/extended_ana_config.yml -output_path ./Scratch/Extended_AnA -np $SLURM_NTASKS
#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/short_range_config.yml -output_path ./Scratch/Short_Range -np $SLURM_NTASKS
#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/short_range_qpf0_config.yml -output_path ./Scratch/Short_Range_QPF0 -np $SLURM_NTASKS
#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/short_range_subhourly_cycling_config.yml -output_path ./Scratch/Short_Range_SubHourly -np $SLURM_NTASKS
#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/medium_range_config.yml -output_path ./Scratch/Medium_Range -np $SLURM_NTASKS
#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/medium_range_blend_config.yml -output_path ./Scratch/Medium_Range_Blend -np $SLURM_NTASKS
#$PYTHON bmi_wrapper.py ./BMI_NextGen_Configs/CONUS/Ensemble_MVP_Config_Files/long_range_mem1_config.yml -output_path ./Scratch/Long_Range -np $SLURM_NTASKS

report-mem

echo -n "End time: "
date

exit 0

##### Options below explained.
## -N, --nodes=<minnodes[-maxnodes]>
##     Request that a minimum of minnodes nodes be allocated to this job.
##     Number of nodes on which to run (uppercase).
## -n, --ntasks=<number>
##     Total number of tasks to run (lowercase).
## -c, --cpus-per-task=<ncpus>
##     Request that ncpus be allocated per process. Number of cpus per task.
## --cores-per-socket=<cores>
##     Restrict  node  selection to nodes with at least the specified
##     number of cores per socket.
## --sockets-per-node=<sockets>
##     Restrict  node selection to nodes with at least the specified
##     number of sockets.
## -t, --time=<time>
##     Set a limit on the total run time of the job or job step.
##     Format: "minutes", "minutes:seconds", "hours:minutes:seconds"
## --mail-type=<types>
##     Send email notification of <types> for job state changes.
## -e, --error=<mode>
##     Specify how stderr is to be redirected. File for batch script's
##     standard error (default same as the stdout).
## -o, --output=<mode>
##     Specify the mode for stdout redirection. File for batch script's
##     standard output (default name is slum-$jobid.out).
## -J, --job-name=<jobname>
##     Specify  a  name for the job.
## -A, --account=<account_name>
##     Charge resources used by this job to specified account.  The account is an arbitrary string.
##     The account name may be changed after job submission using the scontrol command.
##     Account to charge the job, e.g., coastal, ohd.
## -p, --partition=<partition_names>
##     Request a specific partition for the resource allocation.
##     Queue to submit job, e.g., batch, debug.
## --exclusive
##     Run job exclusively, i.e., do not share nodes with other jobs.
#####

