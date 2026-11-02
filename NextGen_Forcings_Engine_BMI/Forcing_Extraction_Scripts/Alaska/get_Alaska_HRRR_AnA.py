import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class AlaskaHRRRAnADownloader(ForecastDownloader):
    """Downloader for Alaska HRRR (AnA) surface forecast data."""
    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start):
        return d_start.hour % 3 == 0

    def get_download_targets(self, _):
        return [1, 2, 3, 4]
