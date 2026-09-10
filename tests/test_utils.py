"""Utilities for ngen-forcing tests."""

import json
import logging
import os
import typing
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

import test_config_classes  # noqa: F401 # Used by test implementations, more convenient to have it in here rather than using more importlib
import test_consts  # noqa: F401 # Used by test implementations, more convenient to have it in here rather than using more importlib
import xarray as xr
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

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.bmi_model import (
    BMIMODEL,
    NWMv3_Forcing_Engine_BMI_model_Base,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
    ConfigOptions,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.consts import TEST_UTILS
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.forcingInputMod import (
    InputForcings,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.geoMod import (
    GeoMeta,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import MpiConfig
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.general_utils import (
    JSON_NOT_SERIALIZABLE_SENTINEL,
    ExpectVsActualError,
    assert_equal_with_tol,
    serialize_to_json,
)

OS_VAR__CREATE_TEST_EXPECT_DATA = "FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA"


def remove_key(input_data: dict, keys_to_exclude: tuple = ()) -> dict:
    """Recursively remove keys from nested dicts.

    Plain keys remove at all levels. Dot-separated keys (e.g. "esmf_field_out._data")
    remove only at the specified path."""
    # Collect keys that apply globally (no dots)
    flat_keys = tuple(k for k in keys_to_exclude if isinstance(k, str) and "." not in k)

    output_data = {}
    for key, val in input_data.items():
        # Skip if this key is in the global exclusion list
        if key in flat_keys:
            continue

        # Example: key="esmf_field_out", keys_to_exclude=("uid64", "esmf_field_out._data", "esmf_field_out.grid")
        # then child_exclude becomes ("_data", "grid") after stripping the "esmf_field_out." prefix
        child_exclude = ()
        if isinstance(key, str):
            child_exclude = tuple(
                path[len(key) + 1 :]  # Strip "key." prefix from the path
                for path in keys_to_exclude
                if isinstance(path, str) and path.startswith(key + ".")
            )

        # Recurse into nested dicts with combined exclusion rules
        if isinstance(val, dict):
            output_data[key] = remove_key(val, flat_keys + child_exclude)
        else:
            output_data[key] = val

    return output_data


def class_to_dict(class_to_convert: typing.Any, keys_to_exclude: list = []) -> dict:
    """Get the attributes of the test class as a dictionary, where the keys are the attribute names and the values are the attribute values.

    This is useful for serializing the test class to JSON for comparison against expected results.
    """
    data = {}
    # parrent_class_dict=self.test_class.__class__.__base__.__dict__
    # child_class_dict=self.test_class.__class__.__dict__
    for key in dir(class_to_convert):
        if key in keys_to_exclude:
            continue

        val = getattr(class_to_convert, key)
        if not callable(val) and not key.startswith("_"):
            if isinstance(val, (MpiConfig, ConfigOptions, InputForcings, GeoMeta)):
                data[key] = remove_key(
                    class_to_dict(val, keys_to_exclude), keys_to_exclude
                )
            elif isinstance(val, dict):
                data[key] = remove_key(val, keys_to_exclude)
            elif isinstance(val, datetime):
                data[key] = val.strftime("%Y-%m-%dT%H:%M:%S")
            elif isinstance(val, xr.Dataset):
                data[key] = val.to_dict()
            else:
                data[key] = val
    return data


def copy_and_stringify_functions(d: dict) -> dict:
    """Copy dict and stringify functions in the dict."""
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            new_dict[key] = copy_and_stringify_functions(value)
        elif callable(value):
            # Convert function to its string representation (e.g., function name)
            new_dict[key] = value.__name__
        else:
            # Keep other values (strings, ints, etc.) as they are
            new_dict[key] = value
    return new_dict


def convert_long_lists(data: typing.Any, max_length: int = 10) -> typing.Any:
    """Recursively iterate over a nested data dictionary and convert all lists longer than max_length to a hash.

    Args:
        data: The data structure to process (dict, list, or other)
        max_length: Maximum list length before conversion (default: 10)

    Returns:
        Modified copy of the data structure

    """
    if max_length is None:
        return data
    if isinstance(data, dict):
        return {
            key: convert_long_lists(value, max_length) for key, value in data.items()
        }
    elif isinstance(data, list):
        if len(data) > max_length:
            if isinstance(data[0], list):
                for i, val in enumerate(data):
                    if len(val) > max_length:
                        data[i] = f"hash_{hash(tuple(val))}_len_{len(val)}"
            else:
                return f"hash_{hash(tuple(data))}_len_{len(data)}"
        else:
            return [convert_long_lists(item, max_length) for item in data]
    else:
        return data


def assert_no_not_serializable_sentinel(json_str: str) -> None:
    """Assert no not serializable sentinel.

    Inspect the provided string and raise an error if it contains
    the sentinel indicating that it contains objects that could not be serialized to JSON.
    """
    if not isinstance(json_str, str):
        raise TypeError(f"Expected type str for json_str, but got {type(json_str)}")
    if JSON_NOT_SERIALIZABLE_SENTINEL in json_str:
        msg_bookend = f"ERROR: found sentinel JSON_NOT_SERIALIZABLE_SENTINEL ({repr(JSON_NOT_SERIALIZABLE_SENTINEL)}) in the string. Please expand the serializer."
        msg_full = f"vvv {msg_bookend} vvv \n\nThe full string is:\n{json_str}\n\n^^^ {msg_bookend} ^^^"
        raise ValueError(msg_full)


@dataclass
class ClassAttrFetcher:
    """Fetach class Attributes.

    Class attribute fetcher, for helping to collect data
    from various in-memory objects in a parameterized way
    when building test results json files.

    The string dunder of this class is used to build a test result data key.

    Parameters
    ----------
        fixture_attr_name:
            The name of the high-level fixture attribute that contains
            the desired child attribute, e.g. "geo_meta"

        child_attr_name:
            The name of the child attribute or dict key to be collected, e.g. "element_ids".

    """

    fixture_attr_name: str
    child_attr_name: str

    @property
    def results_key_name(self) -> str:
        """Get the key name to be used in test results data for this attribute."""
        return f"{self.fixture_attr_name}__{self.child_attr_name}"

    def __str__(self) -> str:
        """Return string representation of the ClassAttrFetcher."""
        return self.results_key_name

    def get(
        self, fixture_instance: typing.Any, serialize_and_deserialize: bool = False
    ) -> typing.Any:
        """Get attribute value.

        From the fixture, fetch the parent class instance,
        and the value of the child attribute, and return that value.

        Args:
        ----
            fixture_instance: the fixture instance which contains the attributes to fetch from
            serialize_and_deserialize: if true, the returned attribute will be serialized to JSON and then deserialized before returned.

        """
        parent = getattr(fixture_instance, self.fixture_attr_name)
        if isinstance(parent, dict):
            child = parent[self.child_attr_name]
        else:
            child = getattr(parent, self.child_attr_name)
        if serialize_and_deserialize:
            child = json.loads(serialize_to_json(child))
        return child


class BMIForcingFixture:
    """Minimal class of classes for running BMI forcing.

    For example usage, see: tests/esmf_regrid/test_esmf_regrid.test_regrid.
    """

    def __init__(self, cfg: TestConfig_Base) -> None:
        """Initialize BMIForcingFixture."""
        self.bmi_model: NWMv3_Forcing_Engine_BMI_model_Base = BMIMODEL[cfg.grid_type]()
        self.bmi_model.initialize_with_params(config_file=cfg.config_file)

        self.bmi_model_values = self.bmi_model._values
        self.mpi_config: MpiConfig = self.bmi_model._mpi_meta
        self.config_options: ConfigOptions = self.bmi_model._job_meta
        self.geo_meta: GeoMeta = self.bmi_model.geo_meta
        self.input_forcing_mod: dict = self.bmi_model._input_forcing_mod
        self.supp_precip_mod: dict = self.bmi_model._supp_pcp_mod

        self.keys_to_check = cfg.keys_to_check
        self.keys_to_exclude = cfg.keys_to_exclude
        self.map_old_to_new_var_names = cfg.map_old_to_new_var_names
        self.test_file_name_prefix = cfg.test_file_name_prefix

    @staticmethod
    def _trim_arrays_to_input_map_output(data_dict: dict) -> None:
        """Trim arrays in data_dict to match input_map_output length.

        Removes unused forcing indices (e.g., LQFRAC at index 8 when include_lqfrac=0).
        Scans all list values in data_dict and trims those that are longer than input_map_output.
        Modifies data_dict in place.

        Args:
            data_dict: Dictionary containing 'input_map_output' and array keys to trim.
        """
        input_map_output = data_dict.get("input_map_output", [])
        if not input_map_output:
            return  # No trimming if we don't have the mapping

        for key, value in data_dict.items():
            if key == "input_map_output" or value is None:
                continue
            # Only trim list values (which were 2D arrays serialized as list of lists)
            if isinstance(value, list) and len(value) > len(input_map_output):
                logging.debug(
                    f"Trimming {key} from {len(value)} to {len(input_map_output)} rows"
                )
                data_dict[key] = value[: len(input_map_output)]


class BMIForcingFixture_Class(BMIForcingFixture):
    """Test fixture for Class-based tests."""

    def __init__(self, cfg: TestConfig_Base) -> None:
        """Initialize BMIForcingFixture_Class.

        Args:
        ----
            cfg: an instance of TestConfig_Base
        """
        super().__init__(cfg)

        self.expected_sub_dir = "test_data/expected_results"
        self.actual_sub_dir = "test_data/actual_results"
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.extra_attrs: tuple[ClassAttrFetcher] = cfg.extra_attrs
        self.keys_no_hash: tuple[str] = cfg.keys_no_hash
        self.keys_to_exclude_at_init: tuple[str] = cfg.keys_to_exclude_at_init

    def deserial_actual(
        self, suffix: str, current_output_step: str = "", write_to_file: bool = True
    ) -> dict:
        """Get the actual metadata results as a deserialized dictionary, including any extra_attrs."""
        deserial_actual = json.loads(
            serialize_to_json(
                copy_and_stringify_functions(self.test_class_as_dict), sort_keys=True
            )
        )
        # order and reverse so private attributes are last
        deserial_actual = OrderedDict(reversed(list(deserial_actual.items())))
        # Save raw values for keys that should not be hashed
        raw_vals = {
            k: deserial_actual[k] for k in self.keys_no_hash if k in deserial_actual
        }
        deserial_actual = convert_long_lists(deserial_actual, 10)
        deserial_actual.update(raw_vals)
        deserial_actual = remove_key(dict(deserial_actual), self.keys_to_exclude)
        # Add any extra attributes to the results
        for ea in self.extra_attrs:
            deserial_actual[ea.results_key_name] = ea.get(
                self, serialize_and_deserialize=True
            )

        self._trim_arrays_to_input_map_output(deserial_actual)

        if write_to_file:
            self.write_json(
                deserial_actual,
                self.actual_results_file_path(suffix, current_output_step),
            )
        return deserial_actual

    def write_json(self, dictionary_to_write: dict, json_path: str) -> None:
        """Write the deserialized results to a JSON file."""
        json_str = serialize_to_json(dictionary_to_write, sort_keys=True)
        with open(json_path, "w") as f:
            f.write(json_str)

    def deserial_expected(self, suffix: str, current_output_step: str = "") -> dict:
        """Get the expected metadata results as a deserialized dictionary."""
        file_path = self.expected_results_file_path(suffix, current_output_step)

        if os.environ.get(OS_VAR__CREATE_TEST_EXPECT_DATA, "").lower() == "true":
            # Dump current results to disk, to save it as "expected" results for later test runs.
            # Should only be used when committing new test results to the repository.
            logging.warning(f"Writing test data: {file_path}")
            deserial_expected = self.deserial_actual(
                suffix, current_output_step, write_to_file=False
            )
            with open(file_path, "w") as f:
                f.write(serialize_to_json(deserial_expected, sort_keys=True))
            # Remove keys that should be excluded from comparison (match read-exclusion)
            deserial_expected = remove_key(deserial_expected, self.keys_to_exclude)
            if self.map_old_to_new_var_names:
                deserial_expected = self.map_old_to_new_variable_names(
                    deserial_expected
                )
            self._trim_arrays_to_input_map_output(deserial_expected)
            return deserial_expected
        else:
            try:
                with open(file_path) as f:
                    deserial_expected = json.load(f)
                if self.map_old_to_new_var_names:
                    deserial_expected = self.map_old_to_new_variable_names(
                        deserial_expected
                    )
                self._trim_arrays_to_input_map_output(deserial_expected)
                # Remove keys that should be excluded from comparison (match write-exclusion)
                deserial_expected = remove_key(deserial_expected, self.keys_to_exclude)
                # order and reverse so private attributes are last
                return OrderedDict(reversed(list(deserial_expected.items())))
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Could not find {file_path}. Try running the test using OS var {OS_VAR__CREATE_TEST_EXPECT_DATA}=true first to set up the test results expected data."
                ) from e

    def map_old_to_new_variable_names(self, data: dict) -> dict:
        """Map old variable names to new variable names in the expected results data."""
        data_new_keys = {}
        for key, val in data.items():
            if key in TEST_UTILS["OLD_NEW_VAR_MAP"].keys():
                data_new_keys[TEST_UTILS["OLD_NEW_VAR_MAP"][key]] = val
            else:
                data_new_keys[key] = val
        return data_new_keys

    def after_intitialization_check(self) -> None:
        """Run checks after initialization but before any run has been called.

        This is useful for checking the state of the model immediately after initialization, before any updates have occurred.
        """
        logging.info("Starting after_intitialization_check()...")
        orig = self.keys_to_exclude
        self.keys_to_exclude = orig + self.keys_to_exclude_at_init
        try:
            self.compare(self.deserial_actual("init"), self.deserial_expected("init"))
        finally:
            self.keys_to_exclude = orig

    def compare(self, actual: dict, expected: dict) -> None:
        """Compare actual vs expected results."""
        try:
            assert_equal_with_tol(
                expect=expected,
                actual=actual,
                new_keys_in_actual_ok=True,
            )
        except ExpectVsActualError as e:
            raise RuntimeError(
                f"Unexpected results compared to the expected results json: {e}"
            ) from e

    @property
    def test_class_as_dict(self) -> dict:
        """Get the attributes of the test class as a dictionary, where the keys are the attribute names and the values are the attribute values.

        This is useful for serializing the test class to JSON for comparison against expected results.
        """
        return class_to_dict(self.test_class, self.keys_to_exclude)

    def after_bmi_model_update(self, current_output_step: int) -> None:
        """Run checks after bmi_model.update() has been called.

        Args:
        ----
        current_output_step: The current output step, which can be used to conditionally run different checks on the first step vs subsequent steps, since the first step behaves differently in some ways.

        """
        logging.info("Starting after_bmi_model_update()...")
        self.compare(
            self.deserial_actual("after_update", f"_step_{current_output_step}"),
            self.deserial_expected("after_update", f"_step_{current_output_step}"),
        )

    def after_finalize(self) -> None:
        """Run checks after bmi_model.finalize() has been called."""
        logging.info("Starting after_finalize()...")
        self.compare(
            self.deserial_actual("finalize"), self.deserial_expected("finalize")
        )

    def actual_results_file_path(
        self, suffix: str, current_output_step: str = ""
    ) -> str:
        """Get the file path for the actual metadata results JSON file."""
        return f"{self.test_dir}/{self.actual_sub_dir}/test_actual_{self.test_file_name_prefix}_{suffix}_n{self.mpi_config.size}_rank{self.mpi_config.rank}_{current_output_step}.json"

    def expected_results_file_path(
        self, suffix: str, current_output_step: str = ""
    ) -> str:
        """Get the file path for the expected metadata results JSON file."""
        return f"{self.test_dir}/{self.expected_sub_dir}/test_expected_{self.test_file_name_prefix}_{suffix}_n{self.mpi_config.size}_rank{self.mpi_config.rank}_{current_output_step}.json"


class BMIForcingFixture_GeoMod(BMIForcingFixture_Class):
    """Test fixture for GeoMod tests."""

    def __init__(self, cfg: TestConfig_GeoMod) -> None:
        """Initialize BMIForcingFixture_GeoMod.

        Args:
            cfg: an instance of TestConfig_GeoMod
        """
        super().__init__(cfg)
        self.test_class = self.geo_meta


class BMIForcingFixture_ConfigOptions(BMIForcingFixture_Class):
    """Test fixture for ConfigOptions tests."""

    def __init__(self, cfg: TestConfig_ConfigOptions) -> None:
        """Initialize BMIForcingFixture_ConfigOptions.

        Args:
            cfg: an instance of TestConfig_ConfigOptions
        """
        super().__init__(cfg)
        self.test_class = self.config_options


class BMIForcingFixture_InputForcing(BMIForcingFixture_Class):
    """Test fixture for InputForcing tests."""

    def __init__(self, cfg: TestConfig_InputForcing) -> None:
        """Initialize BMIForcingFixture_InputForcing.

        Args:
            cfg: an instance of TestConfig_InputForcing
        """
        super().__init__(cfg)
        self.force_key = cfg.force_key
        self.test_class = self.input_forcing_mod[self.force_key]


class BMIForcingFixture_SuppPrecip(BMIForcingFixture_Class):
    """Test fixture for SuppPrecip tests."""

    def __init__(self, cfg: TestConfig_SuppPrecip) -> None:
        """Initialize BMIForcingFixture_SuppPrecip.

        Args:
            cfg: an instance of TestConfig_SuppPrecip
        """
        super().__init__(cfg)
        self.force_key = cfg.force_key
        self.test_class = self.supp_precip_mod[self.force_key]


class BMIForcingFixture_AnA(BMIForcingFixture_Class):
    """Test fixture for Analysis and Assimilation tests."""

    def __init__(self, cfg: TestConfig_AnA) -> None:
        """Initialize BMIForcingFixture_AnA.

        Args:
            cfg: an instance of TestConfig_AnA
        """
        super().__init__(cfg)
        self.test_class = self.bmi_model._output_obj


class BMIForcingFixture_BmiModel(BMIForcingFixture_Class):
    """Test fixture for BMI model tests."""

    def __init__(self, cfg: TestConfig_BmiModel) -> None:
        """Initialize BMIForcingFixture_AnA.

        Args:
            cfg: an instance of TestConfig_AnA
        """
        super().__init__(cfg)
        self.test_class = self.bmi_model


class BMIForcingFixture_Regrid(BMIForcingFixture):
    def __init__(self, cfg: TestConfig_Regrid) -> None:
        """Writers of regrid tests must call the methods in this order. This is enforced by state attributes.

            self.pre_regrid()
            self.run_regrid()
            self.check_regrid_results()
            self.post_regrid()

        Args:
            cfg: An instance of TestConfig_Regrid
        """
        assert isinstance(cfg, TestConfig_Regrid)
        super().__init__(cfg)

        self.regrid_func = cfg.regrid_func
        self.regrid_arrays_to_trim_extra_elements = (
            cfg.regrid_arrays_to_trim_extra_elements
        )
        self.force_key = cfg.force_key
        self.extra_attrs: tuple[ClassAttrFetcher] = cfg.extra_attrs

        self.cull_force_keys_not_used_this_test()

        self._state = None  # Test fixture state used to help ensure things happen in the right order

    def cull_force_keys_not_used_this_test(self) -> None:
        """Remove force keys that are not used during this test.

        For example, Short Range contains 2 total force keys, one for HRRR and one for RAP,
        but we only want to test one at a time, so remove the other one.
        """
        tmp = {k: v for k, v in self.input_forcing_mod.items() if k == self.force_key}
        if len(tmp) != 1:
            raise ValueError(
                f"Expected to have 1 key-pair in the new input_forcing_mod after culling, but got {len(tmp)}. Original: {self.input_forcing_mod}"
            )
        self.input_forcing_mod = tmp

        tmp = [_ for _ in self.config_options.input_forcings if _ == self.force_key]
        if len(tmp) != 1:
            raise ValueError(
                f"Expected to have 1 element in the proposed new config_options.input_forcings after culling, but got {len(tmp)}. Original: {self.config_options.input_forcings}"
            )
        self.config_options.input_forcings = tmp

    @property
    def serialized_file_suffix(self) -> str:
        """Suffix for the file name for expected test results."""
        gpkg_basename = os.path.splitext(
            os.path.basename(self.config_options.geopackage)
        )[0]
        start_time_str = self.config_options.b_date_proc.strftime("%Y%m%d%H%M%S")
        return f"__{gpkg_basename}_start{start_time_str}_n{self.mpi_config.size}_rank{self.mpi_config.rank}_timestep{self.config_options.bmi_time_index}"

    @property
    def regrid_results_file_name_expect(self) -> str:
        """File name for expected test results."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        file_basename = f"test_expect_{self.test_file_name_prefix}{self.regrid_func.__name__}{self.serialized_file_suffix}.json"
        file_path = os.path.join(
            test_dir, "test_data", "expected_results", file_basename
        )
        return file_path

    @property
    def regrid_results_file_name_actual(self) -> str:
        """File name for actual test results."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        file_basename = f"test_actual_{self.test_file_name_prefix}{self.regrid_func.__name__}{self.serialized_file_suffix}.json"
        file_path = os.path.join(test_dir, "test_data", "actual_results", file_basename)
        return file_path

    def pre_regrid(self) -> None:
        """Run various timing setup methods and preprocessing steps needed *before*  each regrid call."""
        if self._state not in (None, "post_ran"):
            raise ValueError(
                f"In pre_regrid, expected state to be either None or 'post_ran' but got {repr(self._state)}. The test is set up incorrectly."
            )

        config_options = self.config_options
        mpi_config = self.mpi_config
        geo_meta = self.geo_meta
        supp_pcp_mod = self.bmi_model._supp_pcp_mod
        output_obj = self.bmi_model._output_obj
        input_forcing_mod = self.bmi_model._input_forcing_mod

        future_time = (
            self.bmi_model._values["current_model_time"]
            + self.bmi_model._values["time_step_size"]
        )
        model = self.bmi_model._model

        ### NOTE with the exception of setting the skip flag, the below
        ### block is copied verbatim from NWMv3ForcingEngineModel.run()
        (
            future_time,
            config_options,
        ) = model.determine_forecast(
            future_time,
            config_options,
        )
        (
            config_options,
            input_forcing_mod,
            mpi_config,
        ) = model.adjust_precip(
            config_options,
            input_forcing_mod,
            mpi_config,
        )
        (
            config_options,
            mpi_config,
        ) = model.log_forecast(
            config_options,
            mpi_config,
        )
        ### NOTE setting the flag causes the regrid step to be skipped
        self.set_input_forcings_skip_flags()
        (
            future_time,
            config_options,
            geo_meta,
            input_forcing_mod,
            supp_pcp_mod,
            mpi_config,
            output_obj,
            input_forcings,
        ) = model.loop_through_forcing_products(
            future_time,
            config_options,
            geo_meta,
            input_forcing_mod,
            supp_pcp_mod,
            mpi_config,
            output_obj,
        )

        # Update test fixture status
        self._state = "pre_ran"

    def set_input_forcings_skip_flags(self) -> None:
        """Set the `skip` flag on the InputForcings object so that forcing regrid will not occur during loop_through_forcing_products()."""
        logging.debug(
            "Setting input_forcing.skip = True for each value in dict self.input_forcing_mod"
        )
        for force_key, input_forcing in self.bmi_model._input_forcing_mod.items():
            input_forcing.skip = True

    def run_regrid(self, arg1: typing.Any) -> None:
        """Run the regrid function.

        Args:
        ----
            arg1: The first argument to the regrid function, which can vary.
            For example is may be `input_forcings`, or `supplemental_precip`, or potentially others.
            Subsequent arguments to the regrid function should be standard and do not need to be provided by the test caller.

        """
        if self._state != "pre_ran":
            raise ValueError(
                f"In run_regrid, expected state to 'pre_ran' but got {repr(self._state)}. The test is set up incorrectly."
            )

        # regrid_inputs_json_str = serialize_to_json(
        #     arg1,
        #     out_file=f"tmp_regrid_inputs{self.serialized_file_suffix}.json",
        #     sort_keys=True,
        # )
        # geo_meta_json_str = serialize_to_json(
        #     self.geo_meta,
        #     out_file=f"tmp_geo_meta{self.serialized_file_suffix}.json",
        #     sort_keys=True,
        # )

        logging.info(
            f"Calling regrid function: {self.regrid_func.__name__} using arg1 of type {type(arg1)}"
        )
        self.regrid_func(
            arg1,
            config_options=self.config_options,
            wrf_hydro_geo_meta=self.geo_meta,
            mpi_config=self.mpi_config,
        )
        logging.info(f"Done calling regrid function: {self.regrid_func.__name__}")
        # Update test fixture status
        self._state = "regrid_ran"

    def remove_extra_data_from_regrid_results(
        self, input_forcings: InputForcings
    ) -> dict:
        """Validate some high-level aspects of the InputForcings object, such as length and sequence of some arrays.

        Then build a dictionary equivalent of it, and trim some of the arrays to the needed size for tests, and return that dictionary.

        Resulting output numerical arrays of regridding process may contain extra elements that are
        unused, contain unpredictable values, and should be ignored, i.e. should be removed from test results.

        This is detected by inspecting lengths and inspecting explicit array index positions
        referenced by `input_map_output`.

        For example:
            `input_map_output` may contain 8 elements spanning values 0 through 7 (in any order), while `regridded_forcings1` and `regridded_forcings2` may contain 9 elements each.
            In this case, we infer that index 8 (the ninth element) should be ignored.
            We assert that this is the case by confirming that index 8 does not exist in `input_map_output`.
            Then we remove that element from the right end of `regridded_forcings1` and `regridded_forcings2`.

        Args:
        ----
            input_forcings: The InputForcings object immediately after a ESMF regridding has occurred.

        Returns:
        -------
            A dictionary representation of input_forcings, but with some arrays trimmed and some keys dropped.

        """
        ### This is returned after being modified.
        input_forcings_deserial = json.loads(
            serialize_to_json(class_to_dict(input_forcings, self.keys_to_exclude))
        )
        ### e.g. ['TMP_2maboveground', 'SPFH_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground', 'APCP_surface', 'DSWRF_surface', 'DLWRF_surface', 'PRES_surface']
        netcdf_var_names = input_forcings.netcdf_var_names
        ### e.g. ['TMP', 'SPFH', 'UGRD', 'VGRD', 'APCP', 'DSWRF', 'DLWRF', 'PRES']
        grib_vars = input_forcings.grib_vars
        ### The order that the vars appear in the regridded output numerical arrays. e.g. [4, 5, 0, 1, 3, 7, 2, 6] means that "TMP" is at index 4 (the fifth item) in the numerical array.
        input_map_output = input_forcings.input_map_output

        errors: list[Exception] = []

        ### Assert that input_map_output has no duplicates
        if len(input_map_output) != len(set(input_map_output)):
            errors.append(
                ValueError(f"Duplicates exist in input_map_output: {input_map_output}")
            )

        ### Assert that input_map_output has no gaps
        for i in range(len(input_map_output)):
            if i not in input_map_output:
                errors.append(
                    ValueError(
                        f"Index {i} is missing from input_map_output: {input_map_output}"
                    )
                )

        ### Assert that the range of input_map_output is sequential from 0
        if min(input_map_output) != 0:
            errors.append(
                ValueError(
                    f"Expected min(input_map_output) to be 0 but got {min(input_map_output)}"
                )
            )
        if max(input_map_output) != len(input_map_output) - 1:
            errors.append(
                ValueError(
                    f"Expected max(input_map_output) to be (len(input_map_output) - 1) aka ({len(input_map_output) - 1}) but got {max(input_map_output)}"
                )
            )

        ### Assert that netcdf_var_names and grib_vars have equal length
        if len(netcdf_var_names) != len(grib_vars):
            errors.append(
                ValueError(
                    f"len(netcdf_var_names) != len(grib_vars): {len(netcdf_var_names)} vs {len(grib_vars)}"
                )
            )

        ### Remove extra indexes from output arrays.
        for key in self.regrid_arrays_to_trim_extra_elements:
            logging.debug(
                f"Trimming values of key {key} if they are longer than input_map_output (if longer than {len(input_map_output)})"
            )
            attr = input_forcings_deserial[key]
            ### Some are naturally None.
            if attr is None:
                continue
            ### If equal length already, then there's nothing to do.
            if len(attr) == len(input_map_output):
                continue
            ### Assert that length of the array is at least as long as input_map_output.
            if len(attr) < len(input_map_output):
                errors.append(
                    ValueError(
                        f"Expected length of array for key {key} to be at least as long as input_map_output ({len(input_map_output)}), but got length {len(attr)}"
                    )
                )
                continue
            ### Trim the 2d list. First assert it is a list of lists (it should have originated as a 2D array)
            if not isinstance(attr, list):
                errors.append(
                    f"Expected type list[list], got {type(attr)} for key {key}"
                )
                continue
            if not isinstance(attr[0], list):
                errors.append(
                    f"Expected type list[list], got {type(attr)} for key {key}"
                )
                continue

        if errors:
            raise RuntimeError(f"input_forcings had invalid state. Errors: {errors}")

        ### After validation, trim the arrays to the appropriate length.
        self._trim_arrays_to_input_map_output(input_forcings_deserial)

        return input_forcings_deserial

    def check_regrid_results(self, input_forcings: InputForcings) -> None:
        """Check the regrid results against previously serialized expected results data, which should be in the repository.

        Run this with a certain OS var to set up fresh test results expected data files.

        Args:
        ----
            input_forcings: The InputForcings object immediately after a ESMF regridding has occurred.

        """
        if self._state != "regrid_ran":
            raise ValueError(
                f"In check_regrid_results, expected state to 'regrid_ran' but got {repr(self._state)}. The test is set up incorrectly."
            )

        ### Check the output regrid weights file
        pass  # TODO is this needed?

        ### Convert the raw input_forcings object to a serialized-then-deserialized dictionary, trimming some of the arrays as needed.
        regrid_results_actual = self.remove_extra_data_from_regrid_results(
            input_forcings
        )

        ### Add extra results keys from objects outside of InputForcings,
        ### for example this is used to bring in GeoMetaWrfHydro.element_ids.
        for ea in self.extra_attrs:
            if ea.results_key_name in regrid_results_actual:
                raise KeyError(
                    f"Key {ea.results_key_name} already exists in {regrid_results_actual}"
                )
            regrid_results_actual[ea.results_key_name] = ea.get(
                self, serialize_and_deserialize=True
            )

        ### Remove unchecked keys
        regrid_results_actual = {
            k: v for k, v in regrid_results_actual.items() if k in self.keys_to_check
        }
        # regrid_results_actual["nx_local"] = float("inf")  # This will trigger a test failure
        # regrid_results_actual["ny_local"] = float("-inf")  # This will trigger a test failure
        ### Serialize the cleaned up dictionary
        regrid_results_json_str = serialize_to_json(
            regrid_results_actual, sort_keys=True
        )
        assert_no_not_serializable_sentinel(regrid_results_json_str)

        logging.warning(f"Writing actual data: {self.regrid_results_file_name_actual}")
        with open(self.regrid_results_file_name_actual, "w") as f:
            f.write(regrid_results_json_str)

        ### NOTE this should be rarely used, only when updating the test expected outputs dataset
        if os.environ.get(OS_VAR__CREATE_TEST_EXPECT_DATA, "").lower() == "true":
            # Dump current results to disk, to save it as "expected" results for later test runs.
            # Should only be used when committing new test results to the repository.
            logging.warning(
                f"Writing test data: {self.regrid_results_file_name_expect}"
            )
            with open(self.regrid_results_file_name_expect, "w") as f:
                f.write(regrid_results_json_str)

        ### Load expected regrid outputs
        logging.info(
            f"Reading expected test results data: {self.regrid_results_file_name_expect}"
        )
        try:
            with open(self.regrid_results_file_name_expect) as f:
                regrid_results_expect = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find {self.regrid_results_file_name_expect}. Try running the test using OS var {OS_VAR__CREATE_TEST_EXPECT_DATA}=true first to set up the test results expected data."
            ) from e

        # This handles ignoring extra elements that shouldn't be checked which might contain random values, e.g. when include_lqfrac=0.
        self._trim_arrays_to_input_map_output(regrid_results_expect)

        # keys_to_check = ("nx_local", "ny_local", "nx_global", "ny_global")
        # regrid_results_expect = {
        #     k: v for k, v in regrid_results_expect.items() if k in keys_to_check
        # }

        try:
            assert_equal_with_tol(
                expect=regrid_results_expect,
                actual=regrid_results_actual,
                keys_to_check=self.keys_to_check,
            )
        except ExpectVsActualError as e:
            raise RuntimeError(
                f"Unexpected results compared to {self.regrid_results_file_name_expect}: {e}"
            ) from e

    def post_regrid(self) -> None:
        """Run various timing setup methods and postprocessing steps needed *after* each regrid call."""
        if self._state != "regrid_ran":
            raise ValueError(
                f"In post_regrid, expected state to 'regrid_ran' but got {repr(self._state)}. The test is set up incorrectly."
            )

        # Manually update some timing attributes, following existing conventions. (20260227)
        # From bottom of model.py run()
        self.config_options.bmi_time_index += 1
        # From bmi_model.py update_until()
        self.bmi_model._values["current_model_time"] += self.bmi_model._values[
            "time_step_size"
        ]
        # Update test fixture status
        self._state = "post_ran"
