import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class AlaskaHRRRDownloader(ForecastDownloader):
    """
    Downloader for Alaska HRRR (AnA) surface forecast data.

    - Files are stored under `hrrr.YYYYMMDD/alaska/` on the s3 server.
    - Files are available hourly.
    - No forecast hours – just one analysis file per hour.
    """

    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start):
        return d_start.hour % 3 == 0

    def get_download_targets(self, d_start):
        # Forecast hours vary depending on cycle
        if d_start.hour % 3 == 0:
            return range(0, 49) if d_start.hour == 0 else range(0, 19)
        else:
            return []  # Skip non-forecast cycles

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, "hrrr." + d_start.strftime('%Y%m%d'), "alaska")

    def build_file_url_and_name(self, d_start, fhr, _):
        """
        Alaska HRRR files use .ak.grib2 extension and are in the /alaska/ folder.
        """
        fhr_str = str(fhr).zfill(2)
        filename = f"hrrr.t{d_start.strftime('%H')}z.wrfsfcf{fhr_str}.ak.grib2"
        date_path = "hrrr." + d_start.strftime('%Y%m%d')
        url = os.path.join(self.base_url, date_path, "alaska", filename)
        return url, filename


if __name__ == "__main__":
    downloader = AlaskaHRRRDownloader.from_cli_args()
    downloader.run()
