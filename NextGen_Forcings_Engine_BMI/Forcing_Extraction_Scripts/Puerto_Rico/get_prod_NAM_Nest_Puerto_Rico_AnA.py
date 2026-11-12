from Forcing_Extraction_Scripts.forecast_download_base import NAMNestAnADownloader


class NAMNestAnAPuertoRicoDownloader(NAMNestAnADownloader):
    """Downloader for NAM Nest Puerto Rico 3-km forecasts (AnA): hours 1-6 of the 6-hourly cycle containing each hour."""

    domain = "priconest"


if __name__ == "__main__":
    NAMNestAnAPuertoRicoDownloader.cli_main()
