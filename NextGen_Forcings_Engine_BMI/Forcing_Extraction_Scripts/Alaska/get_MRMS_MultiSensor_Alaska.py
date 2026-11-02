from abc import ABC
from Forcing_Extraction_Scripts.forecast_download_base import FixedFileDownloader

class MRMSMultiSensorAlaskaDownloader(FixedFileDownloader, ABC):
    """Downloader for MRMS MultiSensor QPE files for Alaska (Pass1 and Pass2)."""

    @property
    def base_url(self):
        return "https://noaa-mrms-pds.s3.amazonaws.com/ALASKA"

    def build_output_dir(self, _, __):
        return self.out_dir
