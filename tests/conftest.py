"""Conventional pytest file conftest.py. Automatically discovered and implicitly imported by pytest."""

import pytest
from test_utils import (
    BMIForcingFixture,
    BMIForcingFixture_BmiModel,
    BMIForcingFixture_GeoMod,
    BMIForcingFixture_InputForcing,
    BMIForcingFixture_Regrid,
)

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.bmi_model import (
    BMIMODEL,
    NWMv3_Forcing_Engine_BMI_model,
)


@pytest.fixture
def bmi_forcing_fixture(request) -> BMIForcingFixture:
    """Construct minimal class of classes for running BMI forcing.

    Constructor for minimal class of classes for running BMI forcing.
    For example usage, see: tests/esmf_regrid/test_esmf_regrid.test_regrid.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.

    """
    (config_file,) = request.param
    bmi_model = NWMv3_Forcing_Engine_BMI_model()
    bmi_model.initialize_with_params(
        config_file=config_file,
        b_date=None,
        geogrid=None,
        output_path=None,
    )
    return BMIForcingFixture(bmi_model=bmi_model)


@pytest.fixture
def bmi_forcing_fixture_regrid(
    request,
) -> BMIForcingFixture_Regrid:
    """Construct minimal class of callas for running forcing ESMF regrid functions.

    Constructor for minimal class of classes for running forcing ESMF regrid functions.
    For example usage, see: tests/esmf_regrid/test_esmf_regrid.test_regrid.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.

    """
    (
        regrid_func,
        config_file,
        force_key,
        extra_attrs,
        regrid_arrays_to_trim_extra_elements,
        keys_to_check,
        keys_to_exclude,
        grid_type,
    ) = request.param

    bmi_model = BMIMODEL[grid_type]()
    bmi_model.initialize_with_params(
        config_file=config_file,
        b_date=None,
        geogrid=None,
        output_path=None,
    )
    return BMIForcingFixture_Regrid(
        bmi_model=bmi_model,
        regrid_func=regrid_func,
        force_key=force_key,
        keys_to_exclude=keys_to_exclude,
        extra_attrs=extra_attrs,
        regrid_arrays_to_trim_extra_elements=regrid_arrays_to_trim_extra_elements,
        keys_to_check=keys_to_check,
    )


@pytest.fixture
def bmi_forcing_fixture_geomod(
    request,
) -> BMIForcingFixture_GeoMod:
    """Construct minimal class of classes for running forcing GeoMod.

    Constructor for minimal class of classes for running forcing GeoMod.

    For example usage, see: tests/geomod/test_geomod.test_geomod.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.

    """
    (
        config_file,
        keys_to_check,
        keys_to_exclude,
        grid_type,
    ) = request.param

    bmi_model = BMIMODEL[grid_type]()
    bmi_model.initialize_with_params(
        config_file=config_file,
        b_date=None,
        geogrid=None,
        output_path=None,
    )
    return BMIForcingFixture_GeoMod(
        bmi_model=bmi_model,
        keys_to_check=keys_to_check,
        keys_to_exclude=keys_to_exclude,
    )


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--map_old_to_new_var_names",
        action="store",
        default=True,
        help="Argument to specify if old variables names should be mapped to new variable names.",
    )


@pytest.fixture
def bmi_forcing_fixture_input_forcing(
    request,
) -> BMIForcingFixture_InputForcing:
    """Construct minimal class of class for running forcing input_forcing.

    Constructor for minimal class of classes for running forcing input_forcing.

    For example usage, see: tests/forcing_input/test_forcing_input.test_forcing_input.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from @pytest.mark.parametrize usage elsewhere.

    """
    (
        config_file,
        keys_to_check,
        keys_to_exclude,
        grid_type,
        force_key,
    ) = request.param

    bmi_model = BMIMODEL[grid_type]()
    bmi_model.initialize_with_params(
        config_file=config_file,
        b_date=None,
        geogrid=None,
        output_path=None,
    )
    map_old_to_new_var_names = request.config.getoption("--map_old_to_new_var_names")
    if map_old_to_new_var_names == "True" or map_old_to_new_var_names is True:
        map_old_to_new_var_names = True
    elif map_old_to_new_var_names == "False" or map_old_to_new_var_names is False:
        map_old_to_new_var_names = False
    else:
        raise ValueError(
            f"Unexpected value for arg: map_old_to_new_var_names. Expected True or False; recieved: {map_old_to_new_var_names}"
        )

    return BMIForcingFixture_InputForcing(
        bmi_model=bmi_model,
        keys_to_check=keys_to_check,
        keys_to_exclude=keys_to_exclude,
        force_key=force_key,
        map_old_to_new_var_names=map_old_to_new_var_names,
    )


@pytest.fixture
def bmi_forcing_fixture_bmi_model(
    request,
) -> BMIForcingFixture_BmiModel:
    """Construct minimal class of classes for running BMI model tests.

    Constructor for minimal class of classes for running BMI model tests.

    For example usage, see: tests/bmi_model/test_bmi_model.test_bmi_model.

    Args:
        request: A built-in convention for pytest.fixture.  It may be passed from
            @pytest.mark.parametrize usage elsewhere.

    """
    (
        config_file,
        keys_to_check,
        keys_to_exclude,
        grid_type,
    ) = request.param

    bmi_model = BMIMODEL[grid_type]()
    bmi_model.initialize_with_params(
        config_file=config_file,
        b_date=None,
        geogrid=None,
        output_path=None,
    )
    return BMIForcingFixture_BmiModel(
        bmi_model=bmi_model,
        keys_to_check=keys_to_check,
        keys_to_exclude=keys_to_exclude,
    )
