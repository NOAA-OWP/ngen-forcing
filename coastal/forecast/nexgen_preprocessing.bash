#/usr/bin/evn bash

set -x

nexgen_preprocessing() {
    coastal_script_dir=$1
    coastal_parm_dir=$2
    domain=$3
    hrrrfile=$4
    coastal_work_dir=$5
    hrrrdir=$6
    start_time="$7"
    end_time="$8"
    troutepath=$9


     # for the atlgulf domain, this script takes 4+ hours to finish
    $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/SCHISM/HRRR/CONUS/makeSfluxNcFileHRRR.py \
       $coastal_parm_dir/$domain/hgrid.cpp \
       $coastal_parm_dir/$domain/hgrid.gr3 \
       $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/Coastal_Data/Enclosure_Polygons/SCHISM_Atl_Enclosure.pol \
       $hrrrfile \
       $coastal_work_dir \
       $domain
     #instead, copy the pre-createt file  
    #cp $coastal_parm_dir/$domain/sflux2sourceInput.nc $coastal_work_dir/

    $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/SCHISM/HRRR/CONUS/makeAtmoHRRR.py \
	    $coastal_work_dir/sflux2sourceInput.nc \
       $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/Coastal_Data/Enclosure_Polygons/SCHISM_Atl_Enclosure.pol \
       $hrrrdir \
       $coastal_work_dir \
       $domain  \
       "$start_time" \
       "$end_time"

    mkdir -p $coastal_work_dir/sflux
    ln -s  $coastal_work_dir/sflux_air_1.0001.nc $coastal_work_dir/sflux

    $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/SCHISM/merge_precip_only.py \
	    $coastal_work_dir/sflux2sourceInput.nc \
            $coastal_work_dir/source2.nc \
       $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/Coastal_Data/Enclosure_Polygons/SCHISM_Atl_Enclosure.pol \
       $coastal_work_dir \
       $domain

    python $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/SCHISM/boundary_elements.py \
       $coastal_parm_dir/$domain/hgrid.gr3 \
       $coastal_parm_dir/LowerColoradoRiver_InflowOutflows.csv \
       "$coastal_work_dir/"
    python $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/SCHISM/TRoute_source_sink_only.py \
       "$coastal_work_dir/" \
       "$troutepath" \
       "$start_time" \
       "$end_time"   \
       $domain

    python $coastal_script_dir/Coastal_Modeling_Preprocessing_Scripts/SCHISM/merge_source_sink.py \
       "$coastal_work_dir/" \
       "$coastal_work_dir/sflux2sourceInput.nc" \
       $domain
}
