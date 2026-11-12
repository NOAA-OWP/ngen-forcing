from abc import ABC
from datetime import timedelta
from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader

class MRMS_SubHourlyConusDownloader(FixedFileDownloader, ABC):
    """Downloader for MRMS subhourly 2-minute Precipitation Rate data."""

    @property
    def base_url(self):
        return "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/PrecipRate_00.00"

    def build_output_dir(self, _, __):
        return self.out_dir

    def get_file_specs(self, d_start):
        specs = []
        for i in range(0, 60, 2):
            current_step = d_start + timedelta(minutes=i)
            subdir = f"{d_start.strftime('%Y%m%d')}"
            filename = f"MRMS_PrecipRate_00.00_{d_start.strftime('%Y%m%d')}-{current_step.strftime('%H%M%S')}.grib2.gz"
            specs.append((subdir, filename))
        return specs

if __name__ == "__main__":
    MRMS_SubHourlyConusDownloader.cli_main()
