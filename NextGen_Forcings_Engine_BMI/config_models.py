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
