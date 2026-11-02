import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NAMNestAlaskaDownloader(ForecastDownloader):
    """Downloader for NAM Alaska Nest forecast data (3-km resolution)."""
    default_lookback = 36
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-nam-pds.s3.amazonaws.com"

    def get_download_targets(self, d_start):
        return range(1, 61) if d_start.hour in [0, 6, 12, 18] else []
