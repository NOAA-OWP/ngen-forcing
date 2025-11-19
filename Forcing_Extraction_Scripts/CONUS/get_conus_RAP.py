import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class RAPDownloader(ForecastDownloader):
    """
    Downloader for CONUS RAP forecast data.
    Downloads surface forecast files (awp130bgrbfXX.grib2) for each forecast cycle.
    Forecast length is 39 hours for 6-hour cycles, and 21 hours otherwise.
    """

    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-rap-pds.s3.amazonaws.com"

    def get_download_targets(self, d_start):
        # RAP cycles at 03, 09, 15, 21 UTC produce 51-hour forecasts; others produce 21-hour forecasts
        return range(0, 52) if d_start.hour in [3, 9, 15, 21] else range(0, 22)

    def build_output_dir(self, d_start, _):
        # Output directory format: <out_dir>/rap.YYYYMMDD/
        return os.path.join(self.out_dir, f"rap.{d_start.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        """
        Construct the download URL and filename for a given forecast hour.

        Example filename: rap.t00z.awp130bgrbf01.grib2
        URL format: https://.../rap.YYYYMMDD/rap.tHHz.awp130bgrbfXX.grib2
        """
        fhr_str = str(forecast_hour).zfill(2)
        filename = f"rap.t{d_start.strftime('%H')}z.awp130bgrbf{fhr_str}.grib2"
        url = os.path.join(self.base_url, f"rap.{d_start.strftime('%Y%m%d')}", filename)
        return url, filename


if __name__ == "__main__":
    downloader = RAPDownloader.from_cli_args()
    downloader.run()
