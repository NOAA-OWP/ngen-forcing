from Forcing_Extraction_Scripts.forecast_download_base import MRMSMultiSensorDownloader


class MRMSMultiSensorPuertoRicoDownloader(MRMSMultiSensorDownloader):
    """Downloader for MRMS MultiSensor hourly QPE (Pass1 and Pass2) for Puerto Rico."""

    region = "CARIB"


if __name__ == "__main__":
    MRMSMultiSensorPuertoRicoDownloader.cli_main()
