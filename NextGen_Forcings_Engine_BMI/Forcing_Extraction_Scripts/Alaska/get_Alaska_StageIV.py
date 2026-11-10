from Forcing_Extraction_Scripts.forecast_download_base import FileSpec, FixedFileDownloader

class AlaskaStageIVDownloader(FixedFileDownloader):
    """Downloader for Alaska Stage IV 6-hourly precipitation analysis (flat output directory)."""
    default_lookback = 36
    default_cleanback = 240
    default_lagback = 0
    mutable_upstream = True  # Stage IV analyses are rewritten upstream about 24 h after valid time

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/v4.1"

    def should_process_hour(self, d_start):
        return d_start.hour % 6 == 0

    def get_file_specs(self, d_start):
        filename = f"st4_ak.{d_start.strftime('%Y%m%d%H')}.06h.grb2"
        return [FileSpec(f"pcpanl.{d_start.strftime('%Y%m%d')}", filename)]

if __name__ == "__main__":
    AlaskaStageIVDownloader.cli_main()
