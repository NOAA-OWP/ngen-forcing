from abc import ABC

from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader

from datetime import timedelta

class MRMS_SubHourly_ConusDownloader(FixedFileDownloader, ABC):
    """
    Downloader for MRMS MultiSensor hourly QPE data (Pass1 and Pass2).

    This downloader overrides cleanup and download logic because:
    - There is no forecast-hour loop.
    - We always download two fixed files per hour: one from each Pass.
    """

    @property
    def base_url(self):
        # Root s3 bucket URL to extract 2-minute Precipitation MRMS data
        return "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/PrecipRate_00.00"

    def build_output_dir(self, _, __):
        return self.out_dir

    def get_file_specs(self, d_start):
        specs = []
        print(d_start)
        #lll
        for i in range(0, 60, 2):
            current_step = d_start + timedelta(minutes=i)
            subdir = f"{d_start.strftime('%Y%m%d')}"
            filename = f"MRMS_PrecipRate_00.00_{d_start.strftime('%Y%m%d')}-{current_step.strftime('%H%M%S')}.grib2.gz"
            specs.append((subdir, filename))
        return specs


if __name__ == "__main__":
    downloader = MRMS_SubHourlyConusDownloader.from_cli_args()
    downloader.run()
