import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NAMNestAnAPuertoRicoDownloader(ForecastDownloader):
    """
    Downloader for NAM Nest Puerto Rico 3-km forecasts.

    - Forecasts are issued every 6 hours: 00Z, 06Z, 12Z, 18Z.
    - Files end with `.priconest.hiresfNN.tm00.grib2`.
    """

    @property
    def base_url(self):
        return "https://noaa-nam-pds.s3.amazonaws.com"

    def get_download_targets(self, _):
        return range(1, 7)

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"nam.{d_start.strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start, target, _):
        if d_start.hour in [0, 1, 2, 3, 4, 5]:
            d_start = d_start.replace(hour=0)
        elif d_start.hour in [6, 7, 8, 9, 10, 11]:
            d_start = d_start.replace(hour=6)
        elif d_start.hour in [12, 13, 14, 15, 16, 17]:
            d_start = d_start.replace(hour=12)
        elif d_start.hour in [18, 19, 20, 21, 22, 23]:
            d_start = d_start.replace(hour=18)
        fhr = str(target).zfill(2)
        filename = f"nam.t{d_start.strftime('%H')}z.priconest.hiresf{fhr}.tm00.grib2"
        url = os.path.join(self.base_url, f"nam.{d_start.strftime('%Y%m%d')}", filename)
        return url, filename


if __name__ == "__main__":
    downloader = NAMNestAnAPuertoRicoDownloader.from_cli_args()
    downloader.run()
