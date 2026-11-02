from abc import ABC
from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader

class MRMSMultiSensorHawaiiDownloader(FixedFileDownloader, ABC):
    """Downloader for MRMS MultiSensor QPE (Pass1 and Pass2) for Hawaii."""

    @property
    def base_url(self):
        return "https://noaa-mrms-pds.s3.amazonaws.com/HAWAII"

    def build_output_dir(self, _, __):
        return self.out_dir
