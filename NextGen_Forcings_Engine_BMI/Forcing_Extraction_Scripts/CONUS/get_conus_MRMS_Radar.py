from abc import ABC
from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader

class MRMSRadarConusDownloader(FixedFileDownloader, ABC):
    """Downloader for MRMS RadarOnly hourly QPE for CONUS."""

    @property
    def base_url(self):
        return "https://noaa-mrms-pds.s3.amazonaws.com/"

    def build_output_dir(self, _, __):
        return self.out_dir

    def get_file_specs(self, d_start):
        subdir = f"CONUS/RadarOnly_QPE_01H_00.00/{d_start.strftime('%Y%m%d')}"
        filename = f"MRMS_RadarOnly_QPE_01H_00.00_{d_start.strftime('%Y%m%d')}-{d_start.strftime('%H')}0000.grib2.gz"
        return [(subdir, filename)]

if __name__ == "__main__":
    downloader = MRMSRadarConusDownloader.from_cli_args()
    downloader.run()
