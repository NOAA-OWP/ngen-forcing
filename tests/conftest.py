"""Conventional pytest file conftest.py. Automatically discovered and implicitly imported by pytest."""

import pytest
from test_config_classes import (
    TestConfig_AnA,
    TestConfig_Base,
    TestConfig_BmiModel,
    TestConfig_ConfigOptions,
    TestConfig_GeoMod,
    TestConfig_InputForcing,
    TestConfig_Regrid,
    TestConfig_SuppPrecip,
)
from test_utils import (
    BMIForcingFixture_AnA,
    BMIForcingFixture_BmiModel,
    BMIForcingFixture_ConfigOptions,
    BMIForcingFixture_GeoMod,
    BMIForcingFixture_InputForcing,
    BMIForcingFixture_Regrid,
    BMIForcingFixture_SuppPrecip,
)


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--map_old_to_new_var_names",
        action="store",
        help="Argument to specify if old variables names should be mapped to new variable names.",
    )


def update_cfg_with_cli_inputs(cfg: TestConfig_Base, request) -> None:
    """Update the test config in-place using values passed from CLI.
    Args:
        cfg: An instance of a test config.
        request: pytest convention. May be passed from @pytest.mark.parametrize.
    """
    map_old_to_new_var_names = request.config.getoption("--map_old_to_new_var_names")
    if map_old_to_new_var_names is not None:
        if map_old_to_new_var_names == "True" or map_old_to_new_var_names is True:
            map_old_to_new_var_names = True
        elif map_old_to_new_var_names == "False" or map_old_to_new_var_names is False:
            map_old_to_new_var_names = False
        else:
            raise ValueError(
                f"Unexpected value for arg: map_old_to_new_var_names. Expected True or False; received: {map_old_to_new_var_names}"
            )
        cfg.map_old_to_new_var_names = map_old_to_new_var_names


@pytest.fixture
def bmi_forcing_fixture_regrid(request) -> BMIForcingFixture_Regrid:
    """Construct class for tests of ESMF regrid functions.
    For example usage, see: tests/esmf_regrid/test_esmf_regrid.py.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.
    """
    cfg = request.param
    assert isinstance(cfg, TestConfig_Regrid)
    update_cfg_with_cli_inputs(cfg, request)
    return BMIForcingFixture_Regrid(cfg)


@pytest.fixture
def bmi_forcing_fixture_geomod(request) -> BMIForcingFixture_GeoMod:
    """Construct class for tests of GeoMod.
    For example usage, see: tests/geomod/test_geomod.py.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.
    """
    cfg = request.param
    assert isinstance(cfg, TestConfig_GeoMod)
    update_cfg_with_cli_inputs(cfg, request)
    return BMIForcingFixture_GeoMod(cfg)


@pytest.fixture
def bmi_forcing_fixture_configoptions(request) -> BMIForcingFixture_ConfigOptions:
    """Construct class for tests of ConfigOptions.
    For example usage, see: tests/config_options/test_config_options.py.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.
    """
    cfg = request.param
    assert isinstance(cfg, TestConfig_ConfigOptions)
    update_cfg_with_cli_inputs(cfg, request)
    return BMIForcingFixture_ConfigOptions(cfg)


@pytest.fixture
def bmi_forcing_fixture_input_forcing(request) -> BMIForcingFixture_InputForcing:
    """Construct class for tests of input_forcing.
    For example usage, see: tests/input_forcing/test_input_forcing.py.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.
    """
    cfg = request.param
    assert isinstance(cfg, TestConfig_InputForcing)
    update_cfg_with_cli_inputs(cfg, request)
    return BMIForcingFixture_InputForcing(cfg)


@pytest.fixture
def bmi_forcing_fixture_supp_precip(request) -> BMIForcingFixture_SuppPrecip:
    """Construct class for tests of supp_precip.
    For example usage, see: tests/supp_precip/test_supp_precip.py.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.
    """
    cfg = request.param
    assert isinstance(cfg, TestConfig_SuppPrecip)
    update_cfg_with_cli_inputs(cfg, request)
    return BMIForcingFixture_SuppPrecip(cfg)


@pytest.fixture
def bmi_forcing_fixture_ana(request) -> BMIForcingFixture_AnA:
    """Construct class for tests of Analysis and Assimilation.
    For example usage, see: tests/ana/test_ana.py.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.
    """
    cfg = request.param
    assert isinstance(cfg, TestConfig_AnA)
    update_cfg_with_cli_inputs(cfg, request)
    return BMIForcingFixture_AnA(cfg)


@pytest.fixture
def bmi_forcing_fixture_bmi_model(request) -> BMIForcingFixture_BmiModel:
    """Construct class for running BMI model tests.
    For example usage, see: tests/bmi_model/test_bmi_model.py.

    Constructor for minimal class of classes for running BMI model tests.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.
    """
    cfg = request.param
    assert isinstance(cfg, TestConfig_BmiModel)
    update_cfg_with_cli_inputs(cfg, request)
    return BMIForcingFixture_BmiModel(cfg)
