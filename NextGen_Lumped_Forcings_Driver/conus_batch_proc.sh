#!/bin/bash

# Define the directory containing the .gpkg files
HYDROFAB_DIR="/ngen-app/data/conus_test/"

# Check if the directory exists
if [ ! -d "$HYDROFAB_DIR" ]; then
	    echo "Error: Directory $HYDROFAB_DIR not found"
	        exit 1
	fi

# Count total files for progress tracking
total_files=$(ls ${HYDROFAB_DIR}/*.gpkg | wc -l)
current_file=0

# Process each .gpkg file
for file in ${HYDROFAB_DIR}/*.gpkg
do
    current_file=$((current_file + 1))
    echo "Processing file $current_file of $total_files: $(basename $file)"
    python conus_run.py -W ignore::FutureWarning -W ignore::UserWarning "$file"
done

echo "Processing complete!"

