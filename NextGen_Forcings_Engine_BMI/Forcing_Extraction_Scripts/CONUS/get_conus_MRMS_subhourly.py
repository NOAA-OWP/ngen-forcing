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
