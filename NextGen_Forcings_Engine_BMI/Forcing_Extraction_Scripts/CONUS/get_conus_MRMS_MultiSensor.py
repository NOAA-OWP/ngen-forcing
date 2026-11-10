from Forcing_Extraction_Scripts.forecast_download_base import MRMSMultiSensorDownloader


class MRMSMultiSensorConusDownloader(MRMSMultiSensorDownloader):
    """Downloader for MRMS MultiSensor hourly QPE (Pass1 and Pass2) for CONUS."""

    region = "CONUS"


if __name__ == "__main__":
    MRMSMultiSensorConusDownloader.cli_main()
