import os
import shutil
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class AlaskaStageIVDownloader(ForecastDownloader):
    """Downloader for Alaska Stage IV precipitation analysis."""
    default_lookback = 36
    default_cleanback = 240
    default_lagback = 0

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/v4.1"

    def should_process_hour(self, d_start):
        return d_start.hour % 6 == 0

    def get_download_targets(self, d_start):
        return [0]
