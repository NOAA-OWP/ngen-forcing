import os
import subprocess
from pathlib import Path

def create_hdr_file(dat_file_path):
    """Create a .hdr file for the corresponding .dat file"""
    # Contents from https://nsidc.org/data/user-resources/help-center/how-do-i-convert-snodas-binary-files-geotiff-or-netcdf
    hdr_content = """ENVI
samples = 8192
lines = 4096
bands = 1
header offset = 0
file type = ENVI Standard
data type = 2
interleave = bsq
byte order = 1"""
    
    hdr_path = dat_file_path.with_suffix('.hdr')
    with open(hdr_path, 'w') as f:
        f.write(hdr_content)
    return hdr_path

def is_valid_date(year, month):
    """Check if the year/month combination is valid for processing"""
    try:
        year = int(year)
        month = int(month.split('_')[0])  # Extract numeric part of month
        
        # GDAL settings only valid for data from 01OCT2013 onward
        if year > 2013:
            return True
        elif year == 2013:
            return month > 9
        else:
            return False
    except ValueError:
        return False
def process_dat_file(dat_file_path, base_mount_path):
    """Process a single .dat file - create HDR and convert to NetCDF"""
    # Create HDR file
    create_hdr_file(dat_file_path)
    
    # Setup output path for NetCDF
    year = dat_file_path.parts[-3]  # Get year from path
    output_dir = Path('unmasked') / year / 'NetCDF'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / dat_file_path.name.replace('.dat', '.nc')
    
    # Calculate paths relative to mount point
    relative_input_path = os.path.relpath(dat_file_path, base_mount_path)
    relative_output_path = os.path.relpath(output_file, base_mount_path)
    
    # Docker paths (inside container)
    docker_input_path = f'/data/{relative_input_path}'
    docker_output_path = f'/data/{relative_output_path}'
    
    # Construct docker command
    # GDAL code from https://nsidc.org/data/user-resources/help-center/how-do-i-convert-snodas-binary-files-geotiff-or-netcdf
    docker_cmd = [
        'sudo', 'docker', 'run', '--rm',
        '-v', f'{base_mount_path}:/data',
        'ghcr.io/osgeo/gdal:ubuntu-full-latest',
        'gdal_translate', '-of', 'NetCDF',
        '-a_srs', '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
        '-a_nodata', '-9999',
        '-a_ullr', '-130.51666666666667', '58.23333333333333', '-62.25000000000000', '24.10000000000000',
        docker_input_path, docker_output_path
    ]
    
    try:
        # Execute (python 3.6-safe)
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error processing {dat_file_path}")
            print(f"Command output: {stdout}")
            print(f"Error output: {stderr}")
            return False
            
        print(f"Successfully processed: {dat_file_path}")
        return True
    except Exception as e:
        print(f"Error processing {dat_file_path}: {str(e)}")
        return False

def main():
    # Get absolute path for mounting
    base_mount_path = os.path.abspath('.')
    base_path = Path('.')
    
    # Track statistics
    total_files = 0
    processed_files = 0
    
    # Verify Docker image is available
    try:
        # Execute - python 3.6-safe
        process = subprocess.Popen(
            ['sudo', 'docker', 'image', 'inspect', 'ghcr.io/osgeo/gdal:ubuntu-full-latest'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, _ = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    except subprocess.CalledProcessError:
        print("Docker image not found. Pulling image...")
        subprocess.call(['sudo', 'docker', 'pull', 'ghcr.io/osgeo/gdal:ubuntu-full-latest'])
    
    # Walk through directory structure
    for year_dir in base_path.iterdir():
        if not year_dir.is_dir():
            continue
            
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
                
            # Check if year/month combination is valid
            if not is_valid_date(year_dir.name, month_dir.name):
                print(f"Skipping invalid date: {year_dir.name}/{month_dir.name}")
                continue
            
            # Process all .dat files in the month directory
            for dat_file in month_dir.glob('*.dat'):
                total_files += 1
                if process_dat_file(dat_file, base_mount_path):
                    processed_files += 1
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {total_files - processed_files}")

if __name__ == "__main__":
    main()

