import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NAMNestAnAPuertoRicoDownloader(ForecastDownloader):
    """Downloader for NAM Nest Puerto Rico 3-km forecasts (AnA)."""

    @property
    def base_url(self):
        return "https://noaa-nam-pds.s3.amazonaws.com"

    def get_download_targets(self, _):
        return range(1, 7)

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"nam.{d_start.strftime('%Y%m%d')}")
