from Forcing_Extraction_Scripts.forecast_download_base import FileSpec, FixedFileDownloader


class StageIVDownloader(FixedFileDownloader):
    """Downloader for CONUS Stage IV hourly precipitation analysis (flat output directory)."""
    default_lookback = 36
    default_cleanback = 240
    default_lagback = 0
    mutable_upstream = True  # Stage IV analyses are rewritten upstream about 24 h after valid time

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/v4.1"

    def get_file_specs(self, d_start):
        filename = f"st4_conus.{d_start.strftime('%Y%m%d%H')}.01h.grb2"
        return [FileSpec(f"pcpanl.{d_start.strftime('%Y%m%d')}", filename)]

if __name__ == "__main__":
    StageIVDownloader.cli_main()
