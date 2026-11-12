from Forcing_Extraction_Scripts.forecast_download_base import NAMNestAnADownloader


class NAMNestAnAHawaiiDownloader(NAMNestAnADownloader):
    """Downloader for NAM Nest Hawaii 3-km forecasts (AnA): hours 1-6 of the 6-hourly cycle containing each hour."""

    domain = "hawaiinest"


if __name__ == "__main__":
    NAMNestAnAHawaiiDownloader.cli_main()
