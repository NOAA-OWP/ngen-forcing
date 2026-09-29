"""Module for recreating cached ESMF regrid weights for a gridded configuration.

This was added to support large domains which were consuming too much memory when
the existing ESMF weight-generation operations were called against the full range of variables
for very large domains.

This was added when adding support for coastal workflows.
The original business logic was in the ``nwm-rte`` repo in ``nwm-rte/run_coastal.py`` in the ``coastalforcing` branch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from NextGen_Forcings_Engine.bmi_model import BMIMODEL

from NextGen_Forcings_Engine_BMI.coastal import consts
from NextGen_Forcings_Engine_BMI.coastal.gridded.config_manager import (
    CoastalConfigManager,
)
from NextGen_Forcings_Engine_BMI.run_bmi_model import REFERENCE_TIME_FORMAT

ProductKind = Literal["input", "supplemental"]

LOG = logging.getLogger("FORCING")


@dataclass(frozen=True)
class Product:
    """One configured forcing product to warm in an isolated process."""

    kind: ProductKind
    key: int


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as config_file:
        return yaml.safe_load(config_file)


def _products_to_warm(config: dict[str, Any]) -> list[Product]:
    input_products = [
        Product("input", product_key)
        for product_key in config.get(consts.INPUT_FORCINGS, [])
    ]
    supplemental_products = [
        Product("supplemental", product_key)
        for product_key in config.get(consts.SUPPLEMENTAL_PRECIPITATION, [])
    ]
    return input_products + supplemental_products


def _remove_cached_weights(model, product: Product) -> None:
    forcing_modules = (
        model._input_forcing_mod if product.kind == "input" else model._supp_pcp_mod
    )
    forcing = forcing_modules[product.key]
    file_key = f"{forcing.product_name}_{model._job_meta.geogrid}"
    hash_key = hashlib.md5(file_key.encode()).hexdigest()[:8]
    weight_file = Path(model._job_meta.weightsDir) / f"ESMF_weight_{hash_key}.nc4"
    weight_file.unlink(missing_ok=True)


def _select_product_at_index(
    config: dict[str, Any],
    product_field: str,
    related_fields: tuple[str, ...],
    product_index: int,
) -> None:
    """Keep one entry from each list belonging to a configured product."""
    product_count = len(config[product_field])
    for field in related_fields:
        values = config.get(field)
        if not values:
            continue
        if not isinstance(values, list) or len(values) != product_count:
            raise ValueError(
                f"{field} must contain one entry per {product_field} product"
            )
        config[field] = [values[product_index]]


def _remove_supplemental_products(config: dict[str, Any]) -> None:
    """Remove supplemental products while warming an input forcing."""
    for field in consts.SUPPLEMENTAL_PRODUCT_FIELDS:
        if field in config:
            config[field] = []


def _config_for_product(config: dict[str, Any], product: Product) -> dict[str, Any]:
    """Create a no-output config containing only the product being warmed."""
    product_config = copy.deepcopy(config)
    product_config[consts.OUTPUT] = 0

    if product.kind == "input":
        product_index = product_config[consts.INPUT_FORCINGS].index(product.key)
        _select_product_at_index(
            product_config,
            consts.INPUT_FORCINGS,
            consts.INPUT_PRODUCT_FIELDS,
            product_index,
        )
        _remove_supplemental_products(product_config)
    else:
        if not product_config.get(consts.INPUT_FORCINGS):
            raise ValueError("A supplemental product requires an input forcing")
        _select_product_at_index(
            product_config,
            consts.INPUT_FORCINGS,
            consts.INPUT_PRODUCT_FIELDS,
            0,
        )
        product_index = product_config[consts.SUPPLEMENTAL_PRECIPITATION].index(
            product.key
        )
        _select_product_at_index(
            product_config,
            consts.SUPPLEMENTAL_PRECIPITATION,
            consts.SUPPLEMENTAL_PRODUCT_FIELDS,
            product_index,
        )

    return product_config


def _create_gridded_model(cycle_datetime: datetime, geogrid: Path):
    return BMIMODEL[consts.GRIDDED_GRID_TYPE](
        b_date=cycle_datetime.strftime(REFERENCE_TIME_FORMAT),
        geogrid=str(geogrid),
    )


def _warm_product(
    config_path: Path,
    cycle_datetime: datetime,
    geogrid: Path,
    product: Product,
) -> None:
    config = _config_for_product(_read_config(config_path), product)
    model = None
    initialized = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
        config_file.flush()

        try:
            model = _create_gridded_model(cycle_datetime, geogrid)
            model.initialize(config_file.name)
            initialized = True
            _remove_cached_weights(model, product)
            model.update()
        finally:
            if initialized:
                model.finalize()


def warm_coastal_regrid_weights(
    forcing_configuration: str,
    global_domain: str,
    cycle_datetime: datetime,
    lookback: int | None = None,
    forecast_input_horizons: int | None = None,
    *,
    forcing_root_dir: str | Path = consts.DEFAULT_FORCING_ROOT_DIR,
    forcing_template_dir: str | Path = consts.DEFAULT_FORCING_TEMPLATE_DIR,
    geogrid_dir: str | Path = consts.DEFAULT_GEOGRID_DIR,
    regrid_weights_dir: str | Path = consts.DEFAULT_REGRID_WEIGHTS_DIR,
    spatial_metadata_suffix: Mapping[str, str] = consts.SPATIAL_METADATA_SUFFIX,
    debug_subset_name: str = consts.DEBUG_CONUS_SUBSET,
    debug_conus_subset: bool = False,
) -> None:
    """Build a coastal config and recreate each product's regrid weights."""
    config_manager = CoastalConfigManager(
        forcing_configuration,
        global_domain,
        cycle_datetime=cycle_datetime,
        lookback=lookback,
        forecast_input_horizons=forecast_input_horizons,
        forcing_root_dir=forcing_root_dir,
        forcing_template_dir=forcing_template_dir,
        geogrid_dir=geogrid_dir,
        regrid_weights_dir=regrid_weights_dir,
        spatial_metadata_suffix=spatial_metadata_suffix,
        debug_subset_name=debug_subset_name,
        debug_conus_subset=debug_conus_subset,
    )
    config_path = config_manager.write_config()

    LOG.info(
        "Warming coastal regrid weights: configuration=%s domain=%s",
        forcing_configuration,
        global_domain,
    )
    warm_regrid_weights(
        config_path=config_path,
        cycle_datetime=cycle_datetime,
        geogrid=config_manager.geogrid,
    )


def warm_regrid_weights(
    config_path: str | Path,
    cycle_datetime: datetime,
    geogrid: str | Path,
) -> None:
    """Recreate cached weights for every product, one subprocess per product."""
    config_path = Path(config_path).resolve()
    geogrid = Path(geogrid).resolve()
    config = _read_config(config_path)

    if config.get(consts.GRID_TYPE) != consts.GRIDDED_GRID_TYPE:
        raise ValueError(
            f"Regrid-weight warmup requires {consts.GRID_TYPE}: "
            f"{consts.GRIDDED_GRID_TYPE}"
        )
    if not config.get(consts.REUSE_REGRID_WEIGHTS):
        raise ValueError(
            f"Regrid-weight warmup requires {consts.REUSE_REGRID_WEIGHTS}: true"
        )

    weights_dir = config.get(consts.REGRID_WEIGHTS_DIR)
    if not weights_dir:
        raise ValueError(f"Regrid-weight warmup requires {consts.REGRID_WEIGHTS_DIR}")
    Path(weights_dir).mkdir(parents=True, exist_ok=True)

    products = _products_to_warm(config)
    if not products:
        raise ValueError("The forcing configuration contains no products to warm")

    for product in products:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "NextGen_Forcings_Engine_BMI.coastal.gridded.warm_regrid_weights",
                "--config-path",
                str(config_path),
                "--cycle-datetime",
                cycle_datetime.isoformat(),
                "--geogrid",
                str(geogrid),
                "--product-kind",
                product.kind,
                "--product-key",
                str(product.key),
            ],
            check=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--cycle-datetime", required=True, type=datetime.fromisoformat)
    parser.add_argument("--geogrid", required=True, type=Path)
    parser.add_argument(
        "--product-kind", required=True, choices=("input", "supplemental")
    )
    parser.add_argument("--product-key", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _warm_product(
        args.config_path,
        args.cycle_datetime,
        args.geogrid,
        Product(args.product_kind, args.product_key),
    )


if __name__ == "__main__":
    main()
