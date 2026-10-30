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
