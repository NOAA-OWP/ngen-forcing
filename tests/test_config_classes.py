from __future__ import annotations

import typing
from dataclasses import dataclass, field

if typing.TYPE_CHECKING:
    from test_utils import ClassAttrFetcher


@dataclass(kw_only=True)
class TestConfig_Base:
    """Configuration base class for tests.

    Args:
        test_file_name_prefix: Affects the test data results file names (expected and actual)
        config_file: Forcing configuration file, e.g. "ngen-forcing/tests/test_data/configs/standard_ana_config.yml"
        keys_to_check: The keys to check
        keys_to_exclude: The keys to exclude from the test results json and from equality checks, for example because they contain non-deterministic values or values that are not relevant to the test. Dots in key names are treated as hierarchy separators, so ``"parent.child"`` excludes only ``child`` nested under ``parent`` rather than all keys named ``child``.
        grid_type: e.g. "hydrofabric"
        map_old_to_new_var_names: Whether to map old variable names to new variable names in the expected results data, which is needed when updating the test expected outputs dataset but should be false for regular test runs.
            Note: this is overridden by a CLI arg. See `pytest_addoption` in `conftest.py` for details.
        keys_no_hash: keys whose list values should NOT be hashed, even if long
        keys_to_exclude_at_init: extra keys to exclude only during the init check; supports dot notation like ``keys_to_exclude``
    """

    test_file_name_prefix: str
    config_file: str
    keys_to_check: tuple[str]
    keys_to_exclude: tuple[str]
    grid_type: str
    map_old_to_new_var_names: bool = True
    extra_attrs: list[ClassAttrFetcher] = field(default_factory=list)
    keys_no_hash: tuple[str] = field(default_factory=tuple)
    keys_to_exclude_at_init: tuple[str] = field(default_factory=tuple)


@dataclass(kw_only=True)
class TestConfig_GeoMod(TestConfig_Base):
    """Configuration class for GeoMod Tests

    Args:
        extra_attrs: These are extra attributes to be added to the test results JSON, to supplement the primary InputForcings attributes.
    """


@dataclass(kw_only=True)
class TestConfig_ConfigOptions(TestConfig_Base):
    """Configuration class for ConfigOptions Tests"""


@dataclass(kw_only=True)
class TestConfig_InputForcing(TestConfig_Base):
    """Configuration class for InputForcing Tests.

    Args:
        force_key: Force key to check (often associated with a particular source dataset)
    """

    force_key: int


@dataclass(kw_only=True)
class TestConfig_SuppPrecip(TestConfig_Base):
    """Configuration class for SuppPrecip Tests.

    Args:
        force_key: Force key to check (often associated with a particular source dataset)
    """

    force_key: int


@dataclass(kw_only=True)
class TestConfig_Regrid(TestConfig_Base):
    """Configuration class for Regrid Tests.

    Args:
        force_key: Should agree with the regrid function being tested, e.g. see ginputfunc.forcing_map
        regrid_func: The regrid function that is being tested.
        extra_attrs: These are extra attributes to be added to the test results JSON, to supplement the primary InputForcings attributes.
        regrid_arrays_to_trim_extra_elements: These are output arrays which can contain extra unused elements which need to be removed during an equality check.
    """

    force_key: int
    regrid_func: typing.Callable
    regrid_arrays_to_trim_extra_elements: tuple[str]


@dataclass(kw_only=True)
class TestConfig_AnA(TestConfig_Base):
    """Configuration class for Analysis and Assimilation tests."""


@dataclass(kw_only=True)
class TestConfig_BmiModel(TestConfig_Base):
    """Configuration class for BMI model tests."""
