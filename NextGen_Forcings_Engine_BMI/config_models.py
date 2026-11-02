from __future__ import annotations

import dataclasses
import os
import pathlib
import typing
import yaml


@dataclasses.dataclass
class ConfigOptionsSubset:
    """Subset of Forcings Engine configuration options required for mesh creation."""

    Geopackage: str
    GeogridIn: str
    Parquet_Path: typing.Optional[str] = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        if self.Parquet_Path is not None:
            cleaned_path = str(self.Parquet_Path).strip()
            if not cleaned_path or not os.path.isfile(cleaned_path):
                print(
                    f"NOTICE: Parquet file '{self.Parquet_Path}' is empty or does not exist. "
                    "Setting Parquet_Path to None; bias correction and downscaling methods will not be applied "
                    "in the NextGen Forcings Engine BMI workflow unless a valid parquet file is prescribed."
                )
                self.Parquet_Path = None
            else:
                self.Parquet_Path = cleaned_path

    @classmethod
    def from_yaml_file(cls, path: pathlib.Path | str) -> typing.Self:
        """Instantiate ConfigOptionsSubset directly from a YAML config file."""
        with open(path, "r", encoding="utf-8") as yaml_file:
            loaded_data = yaml.safe_load(yaml_file) or {}
            return cls.from_dict(loaded_data)

    @classmethod
    def from_dict(cls, data: dict[str, typing.Any]) -> typing.Self:
        """Extract and initialize only the fields defined in this dataclass with alias resolution."""
        parquet_val = None
        for key in ("Parquet_Path", "parquet_path", "Parquet", "parquet"):
            if key in data and data[key] is not None:
                parquet_val = data[key]
                break

        data["Parquet_Path"] = parquet_val

        field_names: set[str] = {
            field.name for field in dataclasses.fields(cls) if field.init
        }
        relevant_data: dict[str, typing.Any] = {
            key: value for key, value in data.items() if key in field_names
        }
        return cls(**relevant_data)


@dataclasses.dataclass
class ForcingExtractionConfig:
    """Strongly typed configuration options for operational forcing extraction."""

    b_date_proc: datetime.datetime
    input_forcings: list[typing.Any]
    supp_precip_forcings: list[typing.Any]
    input_force_dirs: list[str]
    supp_precip_dirs: typing.Optional[list[str]]
    fcst_input_horizons: list[int]
    cfsv2EnsMember: typing.Optional[typing.Union[int, str]]
    ana_flag: int
    look_back: int
    max_download_attempts: int = 10
    download_attempt_interval: int = 30
    check_file_availability: int = 0

    @classmethod
    def from_yaml_file(cls, path: pathlib.Path | str) -> typing.Self:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, typing.Any]) -> typing.Self:
        raw_bdate = data.get("RefcstBDateProc")
        b_date = (
            datetime.datetime.strptime(raw_bdate, "%Y-%m-%d %H:%M:%S")
            if isinstance(raw_bdate, str)
            else raw_bdate
        )
        return cls(
            b_date_proc=b_date,
            input_forcings=list(data.get("InputForcings", [])),
            supp_precip_forcings=list(data.get("SuppPcp", [])),
            input_force_dirs=list(data.get("InputForcingDirectories", [])),
            supp_precip_dirs=data.get("SuppPcpDirectories"),
            fcst_input_horizons=list(data.get("ForecastInputHorizons", [])),
            cfsv2EnsMember=data.get("cfsEnsNumber"),
            ana_flag=int(data.get("AnAFlag", 0)),
            look_back=int(data.get("LookBack", 0)),
            max_download_attempts=int(data.get("max_download_attempts", 10)),
            download_attempt_interval=int(data.get("download_attempt_interval", 30)),
            check_file_availability=int(data.get("check_file_availability", 0)),
        )
