import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class HRRRSubhourlyDownloader(ForecastDownloader):
    """
    Downloader for CONUS HRRR sub-hourly data (15-minute forecasts).

    Sub-hourly files follow a naming convention like:
    hrrr.t00z.wrfsubhf00.grib2, ..., wrfsubhf18.grib2

    Always downloads 0–18 forecast hours regardless of cycle time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force a conservative lagback to avoid incomplete subhourly HRRR files
        # This overrides the user-specified --lagBackHours from the CLI
        self._override_lagback = 3

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"

    def should_process_hour(self, d_current):
        return True  # All hours valid for subhourly HRRR

    def get_download_targets(self, d_current):
        return range(0, 19)  # hf00 to hf18

    def build_output_dir(self, d_current):
        return os.path.join(
            self.out_dir,
            f"hrrr.{d_current.strftime('%Y%m%d')}",
            d_current.strftime('%H'),
            "subhourly"
        )

    def build_file_url_and_name(self, d_current, target):
        fhr_str = f"{target:02d}"  # e.g. 00, 01, 02, ...
        hour = d_current.strftime('%H')
        date = d_current.strftime('%Y%m%d')
        filename = f"hrrr.t{hour}z.wrfsubhf{fhr_str}.grib2"
        url = os.path.join(self.base_url, f"hrrr.{date}", hour, filename)
        return url, filename


if __name__ == "__main__":
    downloader = HRRRSubhourlyDownloader.from_cli_args()
    downloader.run()
