from abc import ABC

from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader


class MRMSRadarPuertoRicoDownloader(FixedFileDownloader, ABC):
    """
    Downloader for MRMS RadarOnly hourly QPE for Puerto Rico.

    - Files are stored directly in the RadarOnly_QPE_01H directory.
    - No forecast hours are involved; only one file per timestamp.
    - This subclass overrides cleanup and download logic to reflect
      the simpler structure (one file per hour).
    """

    @property
    def base_url(self):
        # Base directory; files are directly in the RadarOnly_QPE_01H directory
        return "https://noaa-mrms-pds.s3.amazonaws.com/CARIB"

    def build_output_dir(self, _, __):
        return self.out_dir

    def get_file_specs(self, d_start):
        # Construct the filename and subdirectory
        subdir = f"CARIB/RadarOnly_QPE_01H_00.00/{d_start.strftime('%Y%m%d')}"
        filename = f"MRMS_RadarOnly_QPE_01H_00.00_{d_start.strftime('%Y%m%d')}-{d_start.strftime('%H')}0000.grib2.gz"
        return [(subdir, filename)]


if __name__ == "__main__":
    downloader = MRMSRadarPuertoRicoDownloader.from_cli_args()
    downloader.run()
