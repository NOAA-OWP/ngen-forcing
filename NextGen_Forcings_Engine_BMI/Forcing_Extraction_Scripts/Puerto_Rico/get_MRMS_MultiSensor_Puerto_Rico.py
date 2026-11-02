from abc import ABC
from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader

class MRMSMultiSensorPuertoRicoDownloader(FixedFileDownloader, ABC):
    """Downloader for MRMS MultiSensor QPE over Puerto Rico."""

    @property
    def base_url(self):
        return "https://noaa-mrms-pds.s3.amazonaws.com/CARIB/"

    def build_output_dir(self, _, __):
        return self.out_dir
