# Script Descriptions

run_swe.py: This script acts as a wrapper script, coordinating all other mapping scripts in this directory. It accepts and passes all necessary arguments, and it runs each script in a way that enables shared colorbar scaling.

swe_minmax.py: This is a utility script that is not intended to be used in standalone mode. It stores global vmin/vmax values so that each map has the same scale, facilitating comparison. 

convert_swe.py: This script parses multiple catchment-scale .csv output files from ngen, extracting 06z SWE (snow water equivalent) values for the date(s) specified. It writes these values to a NetCDF output file for ease of use with the simulated_swe_mapper.py script. It can be executed in standalone mode, or via the run_swe.py wrapper (recommended). 

simulated_swe_mapper.py: This script reads from a NetCDF file (it assumes a format identical to that created by convert_swe.py), and then plots SWE values on a map for the date specified. Each catchment polygon is filled with the simulated SWE value for that catchment, since these represent lumped values. It can be executed in standalone mode, or via the run_swe.py wrapper (recommended). 

snodas_mapper.py: A mapping script, which plots basin-scale SNODAS SWE values and writes to .png files. It can be executed in standalone mode, or via the run_swe.py wrapper (recommended). 

# Sample data

Some sample data has been included for testing in the sample_data/ directory. This includes a geopackage file and mock .csv files for the corresponding catchments. The .csv files contain manually fabricated SWE values, but are otherwise formatted similarly to actual ngen output. Valid dates for the test data are 2015-12-01 and 2015-12-02. SNODAS SWE values for those dates in the sample basin are > 0. The run_swe.py example below provides a command that will process the sample data.

# Script Usage

A conda environment.yml file has been including. While using a conda environment is optional, this file lists required packages.

#### run_Swe.py:
run_swe.py is expected to be located in the same directory as the other scripts listed here. 

python run_swe.py [-h] date sim_csv_dir sim_netcdf gpkg_file sim_map_output snodas_raw_output snodas_lumped_output

Positional Arguments:  
- date: Date to use for all plots.
- sim_csv_dir: Path that contains ngen swe csv files. This is your ngen output directory.  
- sim_netcdf: Path for simulated swe netcdf file. convert_csv writes to this file, simulated_swe_mapper reads from this file.  
- gpkg_file: Path to geopackage file.  
- sim_map_output: Path where simulated swe map output saved. Output will be a .png file.  
- snodas_raw_output     Path where snodas raw map output saved. Output will be a .png file.
- snodas_lumped_output  Path where snodas catchment map output saved. Output will be a .png file.

#### convert_swe.py: 
python convert_swe.py [-h] csv_directory dates [dates ...] output

Positional Arguments:  
- csv_directory: A string that points to the path that contains the ngen catchment-scale .csv output files to parse.  
- dates: A string representing a date to parse. For example '2015-12-01'. Multiple dates can be entered, but at least one date is required.  
- output: A string representing the full absolute or relative path to desired output file, for example './example.nc'

#### simulated_swe_mapper.py:
python simulated_swe_mapper.py [-h] [--output_file OUTPUT_FILE] [--mode {plot,scan}] netcdf_file gpkg_file date

Arguments:  
- netcdf_file: Required. A string that points to the NetCDF file that contains SWE values. Assumes that the NetCDF file was created by convert_sneqv.py, or has the same format/structure.  
- gpkg_file: Required. A string that points to the .gpkg file containing basin geographic information. (Hydrofabric file).  
- date: Required. A string representing the date you wish to map. Ex: '2015-12-01'  
- output_file: Optional. A string that points to an output file path, ex: './output.png' If no output_file argument is provided, no file will be saved. Instead, the terminal will attempt to display the image using xdg.  
- mode: Optional. Scan mode returns only vmin/vmax, primarily for use in creating a unified color scale. Plot mode is default, and will produce a map image.

#### snodas_mapper.py

python snodas_mapper.py [-h] date gpkg_file output_file_raw output_file_lumped

-h, --help: prints usage

Positional arguments:  
- date: A string representing the date to map. Ex: 2015-12-01  
- gpkg_file: Path to the geopackage file to use.  
- output_file_raw: Path where raw map to be saved.  
- output_file_lumped: Path where lumped map to be saved.

# Examples

#### run_swe.py
python run_swe.py '2015-12-01' ./sample_data/ ./sample_data/test.nc ./sample_data/gages-13240000.gpkg ./sim_map.png ./raw_map.png ./lumped_map.png

#### convert_swe.py
python convert_swe.py -h
python convert_swe.py '/data/ngen_out/01123000/' '2015-12-01' '2015-12-02' '/data/sneqv/01123000_swe.nc'

#### simulated_swe_mapper.py
python simulated_swe_mapper.py -h
python simulated_swe_mapper.py '/data/sneqv/01123000_swe.nc' '/data/geopackages/gages-01123000.gpkg' '2015-12-01' --output_file '/data/maps/swe_20151201_01123000.png'

#### snodas_mapper.py
python snodas_mapper.py -h
python snodas_mapper.py --gpkg_file '/data/geopackages/gages-13240000.gpkg' --output_file '/data/snodas/13240000_c.nc' --plot_type 'catchment' 's3://ngwpc-forcing/snodas_nc/zz_ssm11034tS__T0001TTNATS2009123105HP001.nc'

