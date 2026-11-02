import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class HRRRDownloader(ForecastDownloader):
    """Downloader for CONUS HRRR forecast data."""
    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

    def get_download_targets(self, d_start):
        return range(0, 49) if d_start.hour % 6 == 0 else range(0, 19)
