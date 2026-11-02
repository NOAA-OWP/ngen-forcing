from abc import ABC
from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader

class MRMSRadarConusDownloader(FixedFileDownloader, ABC):
    """Downloader for MRMS RadarOnly hourly QPE for CONUS."""

    @property
    def base_url(self):
        return "https://noaa-mrms-pds.s3.amazonaws.com/"

    def build_output_dir(self, _, __):
        return self.out_dir
