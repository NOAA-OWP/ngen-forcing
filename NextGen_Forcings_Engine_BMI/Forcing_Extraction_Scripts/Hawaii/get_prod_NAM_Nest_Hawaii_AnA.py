import os
from datetime import timedelta
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NAMNestAnAHawaiiDownloader(ForecastDownloader):
    """Downloader for NAM Nest Hawaii 3-km forecasts (AnA): hours 1-6 of the 6-hourly cycle containing each hour."""

    @property
    def base_url(self):
        return "https://noaa-nam-pds.s3.amazonaws.com"

    @staticmethod
    def nam_cycle(d_start):
        """Return the 00/06/12/18Z NAM cycle that covers d_start."""
        return d_start - timedelta(hours=d_start.hour % 6)

    def get_download_targets(self, _):
        return range(1, 7)

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"nam.{self.nam_cycle(d_start).strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start, target, _):
        cycle = self.nam_cycle(d_start)
        fhr = str(target).zfill(2)
        filename = f"nam.t{cycle.strftime('%H')}z.hawaiinest.hiresf{fhr}.tm00.grib2"
        url = os.path.join(self.base_url, f"nam.{cycle.strftime('%Y%m%d')}", filename)
        return url, filename

if __name__ == "__main__":
    NAMNestAnAHawaiiDownloader.cli_main()
