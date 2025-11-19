import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class ARWHawaiiDownloader(ForecastDownloader):
    """
    Downloader for WRF-ARW 2.5 km forecast data over Hawaii.

    - Forecasts available at 00Z and 12Z only.
    - Each forecast extends 48 hours.
    - Files are located under: hiresw.YYYYMMDD/
    """

    @property
    def base_url(self):
        return "https://ftp.ncep.noaa.gov/data/nccf/com/hiresw/prod"

    def should_process_hour(self, d_current):
        return d_current.hour in [0, 12]

    def get_download_targets(self, d_current):
        # Only forecast cycles at 00Z and 12Z
        return range(1, 49)

    def build_output_dir(self, d_current):
        return os.path.join(self.out_dir, f"hiresw.{d_current.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_current, target):
        fhr = str(target).zfill(2)
        filename = f"hiresw.t{d_current.strftime('%H')}z.arw_2p5km.f{fhr}.hi.grib2"
        url = os.path.join(self.base_url, f"hiresw.{d_current.strftime('%Y%m%d')}", filename)
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True


if __name__ == "__main__":
    downloader = ARWHawaiiDownloader.from_cli_args()
    downloader.run()
