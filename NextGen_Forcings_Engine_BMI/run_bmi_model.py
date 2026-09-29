import argparse
import datetime
import pathlib
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import yaml

# This is the NextGen Forcings Engine BMI instance to execute
from NextGen_Forcings_Engine.bmi_model import (
    BMIMODEL,
    NWMv3_Forcing_Engine_BMI_model,
    parse_config,
)

REFERENCE_TIME_FORMAT = "%Y%m%d%H%M"


def get_date_times(start_time: str, end_time: str) -> tuple:
    """Convert start and end time strings to a list of datetimes at hourly intervals."""
    # Convert start and end time from string to datetime
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    return pd.date_range(start=start_time, end=end_time, freq="h"), start_time, end_time


def print_init(model: NWMv3_Forcing_Engine_BMI_model, num_iterations: int) -> None:
    """Print initial information about the model and the number of iterations to run."""
    print(
        f"\nNow looping through {num_iterations} timesteps, updating the model, and extracting forcing data\n"
    )
    print(f"rank: {model._mpi_meta.rank}")
    print(f"grid_type: {model._grid_type}")


def print_pre_update(num, num_iterations, timestamp) -> None:
    """Print the current iteration number and timestamp before updating the model."""
    print("\n---------------------------------------------------")
    print(f"Iteration #{num} of {num_iterations} for timestamp={timestamp}")


def print_post_update(model, start_time) -> None:
    """Extract forcing variables from the model after the update and print summary statistics (max/min) for each variable."""
    # Initialize to None to avoid PyCharm error
    U2D = V2D = LWDOWN = SWDOWN = T2D = Q2D = PSFC = RAINRATE = LQFRAC = CAT_IDS = None
    U2D_NODE = V2D_NODE = LWDOWN_NODE = SWDOWN_NODE = T2D_NODE = Q2D_NODE = (
        PSFC_NODE
    ) = RAINRATE_NODE = None
    U2D_ELEMENT = V2D_ELEMENT = LWDOWN_ELEMENT = SWDOWN_ELEMENT = T2D_ELEMENT = (
        Q2D_ELEMENT
    ) = PSFC_ELEMENT = RAINRATE_ELEMENT = LQFRAC_NODE = LQFRAC_ELEMENT = None

    # ===============================
    # Initialize arrays based on grid type
    # ===============================
    if model._grid_type in {"gridded", "hydrofabric"}:
        varsize = (
            len(model.geo_meta.element_ids_global)
            if model._grid_type == "hydrofabric"
            else model._varsize
        )
        # Shared initialization
        U2D = np.zeros(varsize, dtype=float)
        V2D = np.zeros(varsize, dtype=float)
        LWDOWN = np.zeros(varsize, dtype=float)
        SWDOWN = np.zeros(varsize, dtype=float)
        T2D = np.zeros(varsize, dtype=float)
        Q2D = np.zeros(varsize, dtype=float)
        PSFC = np.zeros(varsize, dtype=float)
        RAINRATE = np.zeros(varsize, dtype=float)
        if model._job_meta.include_lqfrac == 1:
            LQFRAC = np.zeros(varsize, dtype=float)
        if model._grid_type == "hydrofabric":
            CAT_IDS = np.zeros(varsize, dtype=np.int64)

    elif model._grid_type == "unstructured":
        # Unstructured grid (element + node)
        U2D_NODE = np.zeros(model._varsize, dtype=float)
        V2D_NODE = np.zeros(model._varsize, dtype=float)
        LWDOWN_NODE = np.zeros(model._varsize, dtype=float)
        SWDOWN_NODE = np.zeros(model._varsize, dtype=float)
        T2D_NODE = np.zeros(model._varsize, dtype=float)
        Q2D_NODE = np.zeros(model._varsize, dtype=float)
        PSFC_NODE = np.zeros(model._varsize, dtype=float)
        RAINRATE_NODE = np.zeros(model._varsize, dtype=float)

        U2D_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
        V2D_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
        LWDOWN_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
        SWDOWN_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
        T2D_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
        Q2D_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
        PSFC_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
        RAINRATE_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
        if model._job_meta.include_lqfrac == 1:
            LQFRAC_NODE = np.zeros(model._varsize, dtype=float)
            LQFRAC_ELEMENT = np.zeros(model._varsize_elem, dtype=float)
    else:
        raise ValueError(f"Unsupported grid type: {model._grid_type}")

    include_lqfrac = model._job_meta.include_lqfrac == 1
    is_unstructured = model._grid_type == "unstructured"
    bmi_seconds = model.get_current_time()
    model_time = start_time + datetime.timedelta(seconds=bmi_seconds)

    if model._grid_type in {"gridded", "hydrofabric"}:
        if model._grid_type == "hydrofabric":
            CAT_IDS = model.get_value("CAT-ID", CAT_IDS)

        U2D = model.get_value("U2D_ELEMENT", U2D)
        V2D = model.get_value("V2D_ELEMENT", V2D)
        T2D = model.get_value("T2D_ELEMENT", T2D)
        Q2D = model.get_value("Q2D_ELEMENT", Q2D)
        SWDOWN = model.get_value("SWDOWN_ELEMENT", SWDOWN)
        LWDOWN = model.get_value("LWDOWN_ELEMENT", LWDOWN)
        PSFC = model.get_value("PSFC_ELEMENT", PSFC)
        RAINRATE = model.get_value("RAINRATE_ELEMENT", RAINRATE)
        if include_lqfrac:
            LQFRAC = model.get_value("LQFRAC_ELEMENT", LQFRAC)

        values_max = [
            arr.max() for arr in [U2D, V2D, LWDOWN, SWDOWN, T2D, Q2D, PSFC, RAINRATE]
        ]
        values_min = [
            arr.min() for arr in [U2D, V2D, LWDOWN, SWDOWN, T2D, Q2D, PSFC, RAINRATE]
        ]
        if include_lqfrac:
            values_max.append(LQFRAC.max())
            values_min.append(LQFRAC.min())

    else:
        U2D_NODE = model.get_value("U2D_NODE", U2D_NODE)
        V2D_NODE = model.get_value("V2D_NODE", V2D_NODE)
        T2D_NODE = model.get_value("T2D_NODE", T2D_NODE)
        Q2D_NODE = model.get_value("Q2D_NODE", Q2D_NODE)
        SWDOWN_NODE = model.get_value("SWDOWN_NODE", SWDOWN_NODE)
        LWDOWN_NODE = model.get_value("LWDOWN_NODE", LWDOWN_NODE)
        PSFC_NODE = model.get_value("PSFC_NODE", PSFC_NODE)
        RAINRATE_NODE = model.get_value("RAINRATE_NODE", RAINRATE_NODE)

        U2D_ELEMENT = model.get_value("U2D_ELEMENT", U2D_ELEMENT)
        V2D_ELEMENT = model.get_value("V2D_ELEMENT", V2D_ELEMENT)
        T2D_ELEMENT = model.get_value("T2D_ELEMENT", T2D_ELEMENT)
        Q2D_ELEMENT = model.get_value("Q2D_ELEMENT", Q2D_ELEMENT)
        SWDOWN_ELEMENT = model.get_value("SWDOWN_ELEMENT", SWDOWN_ELEMENT)
        LWDOWN_ELEMENT = model.get_value("LWDOWN_ELEMENT", LWDOWN_ELEMENT)
        PSFC_ELEMENT = model.get_value("PSFC_ELEMENT", PSFC_ELEMENT)
        RAINRATE_ELEMENT = model.get_value("RAINRATE_ELEMENT", RAINRATE_ELEMENT)

        if include_lqfrac:
            LQFRAC_NODE = model.get_value("LQFRAC_NODE", LQFRAC_NODE)
            LQFRAC_ELEMENT = model.get_value("LQFRAC_ELEMENT", LQFRAC_ELEMENT)

        values_max = [
            U2D_NODE.max(),
            V2D_NODE.max(),
            LWDOWN_NODE.max(),
            SWDOWN_NODE.max(),
            T2D_NODE.max(),
            Q2D_NODE.max(),
            PSFC_NODE.max(),
            RAINRATE_NODE.max(),
            U2D_ELEMENT.max(),
            V2D_ELEMENT.max(),
            LWDOWN_ELEMENT.max(),
            SWDOWN_ELEMENT.max(),
            T2D_ELEMENT.max(),
            Q2D_ELEMENT.max(),
            PSFC_ELEMENT.max(),
            RAINRATE_ELEMENT.max(),
        ]
        values_min = [
            U2D_NODE.min(),
            V2D_NODE.min(),
            LWDOWN_NODE.min(),
            SWDOWN_NODE.min(),
            T2D_NODE.min(),
            Q2D_NODE.min(),
            PSFC_NODE.min(),
            RAINRATE_NODE.min(),
            U2D_ELEMENT.min(),
            V2D_ELEMENT.min(),
            LWDOWN_ELEMENT.min(),
            SWDOWN_ELEMENT.min(),
            T2D_ELEMENT.min(),
            Q2D_ELEMENT.min(),
            PSFC_ELEMENT.min(),
            RAINRATE_ELEMENT.min(),
        ]
        if include_lqfrac:
            values_max.extend([LQFRAC_NODE.max(), LQFRAC_ELEMENT.max()])
            values_min.extend([LQFRAC_NODE.min(), LQFRAC_ELEMENT.min()])

    print_forcing_summary(
        "max", values_max, include_lqfrac, is_unstructured, model_time
    )
    print_forcing_summary(
        "min", values_min, include_lqfrac, is_unstructured, model_time
    )


def print_forcing_summary(
    label: str,
    values: list[float],
    include_lqfrac: bool,
    is_unstructured: bool,
    model_time: datetime.datetime,
) -> None:
    """Print the summary of forcing variables (max/min) for each timestep.

    :param label: 'max' or 'min', depending on whether we're showing maximum or minimum values.
    :param values: List of values to print.
    :param include_lqfrac: Boolean flag indicating whether to include the liquid fraction of precipitation.
    :param is_unstructured: Boolean flag indicating whether the grid is unstructured.
    :param model_time: The current model time (datetime object) to display with the values.
    """
    print(f"\n==== {label.upper()} VALUES ====")

    model_time_str = model_time.strftime("%Y-%m-%d %H:%M:%S")
    model_time_header = "model time"

    if is_unstructured:
        base_labels = [
            "U2D_NODE",
            "V2D_NODE",
            "LWDOWN_NODE",
            "SWDOWN_NODE",
            "T2D_NODE",
            "Q2D_NODE",
            "PSFC_NODE",
            "RAINRATE_NODE",
            "U2D_ELEMENT",
            "V2D_ELEMENT",
            "LWDOWN_ELEMENT",
            "SWDOWN_ELEMENT",
            "T2D_ELEMENT",
            "Q2D_ELEMENT",
            "PSFC_ELEMENT",
            "RAINRATE_ELEMENT",
        ]
        if include_lqfrac:
            base_labels += ["LQFRAC_NODE", "LQFRAC_ELEMENT"]
    else:
        base_labels = [
            "U2D_ELEMENT",
            "V2D_ELEMENT",
            "LWDOWN_ELEMENT",
            "SWDOWN_ELEMENT",
            "T2D_ELEMENT",
            "Q2D_ELEMENT",
            "PSFC_ELEMENT",
            "RAINRATE_ELEMENT",
        ]
        if include_lqfrac:
            base_labels.append("LQFRAC_ELEMENT")

    # Full headers and row values
    headers = [model_time_header] + base_labels
    value_strings = [model_time_str] + [f"{v:.6g}" for v in values]

    # Determine column widths
    col_widths = [max(len(h), len(v)) for h, v in zip(headers, value_strings)]

    # Print header and value rows
    header_row = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    value_row = " | ".join(f"{v:<{w}}" for v, w in zip(value_strings, col_widths))

    print(header_row)
    print(value_row)


def get_options():
    """Parse command-line arguments for running the BMI model.

    :return: Namespace object containing the parsed arguments.
    """
    # TODO keyword arguments should start with --
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "start_time",
        type=str,
        help="Start time should correspond to the forecast cycle time + 1 timestep. Format = 'YYYY-MM-DD HH:mm:ss'",
    )
    parser.add_argument(
        "end_time",
        type=str,
        help="End time should correspond to the last forecast time step you want to calculate. Format = 'YYYY-MM-DD HH:mm:ss'",
    )
    parser.add_argument(
        "-config_path",
        type=pathlib.Path,
        help="Config path for config.yml, otherwise defaults to ./config.yml",
    )
    parser.add_argument(
        "-b_date",
        type=str,
        help="Begin date, should be the start date/time for the forecast cycle, format= 'YYYYMMDDHHmm'. If omitted, reads from configuration file.",
    )
    parser.add_argument(
        "-geogrid",
        type=str,
        help="Full path for geogrid/ESMF Mesh file. If omitted, reads from configuration file.",
    )
    parser.add_argument(
        "-output_path",
        type=pathlib.Path,
        help="A user-provided output path - must include full directory and filename. If omitted, a filename will be automatically generated, in the ScratchDir specified in the config file.",
    )

    return parser.parse_args()


class ForcingRunner(ABC):
    """Define the shared forcing-engine execution lifecycle."""

    def __init__(
        self,
        config_path: pathlib.Path | None = None,
        cycle_datetime: datetime.datetime | None = None,
        start_time: str | datetime.datetime | None = None,
        end_time: str | datetime.datetime | None = None,
        b_date: str | None = None,
        geogrid: str = None,
        output_path: pathlib.Path = None,
        config: dict | None = None,
        num_updates: int | None = None,
        output_t0: bool = False,
    ) -> None:
        self.config_path = self.resolve_config_path(config_path)
        self.cycle_datetime = cycle_datetime
        self.start_time = start_time
        self.end_time = end_time
        self.b_date = b_date
        self.geogrid = geogrid
        self.output_path = output_path
        self.config = config
        self.requested_num_updates = num_updates
        self.output_t0 = output_t0
        self.model = None
        self.output_steps = None
        self.ngen_datetimes = None
        self.is_general = False
        self.num_updates = None

    @classmethod
    def create(
        cls,
        config_path: pathlib.Path | None = None,
        **kwargs,
    ) -> "ForcingRunner":
        """Create the runner child class appropriate for the configured grid type."""
        config_path = cls.resolve_config_path(config_path)
        config = cls._read_config(config_path)

        if config["GRID_TYPE"] == "gridded":
            child = ForcingRunnerGridded
        else:
            child = ForcingRunnerGeneral

        return child(config_path, config=config, **kwargs)

    @staticmethod
    def resolve_config_path(config_path: pathlib.Path | None) -> pathlib.Path:
        """Resolve a supplied config path or use the bundled default."""
        if config_path is not None:
            return pathlib.Path(config_path)
        return pathlib.Path(__file__).parent.resolve() / "config.yml"

    @staticmethod
    def _read_config(config_path: pathlib.Path) -> dict:
        """Read and parse a forcing configuration file."""
        with config_path.open("r") as config_file:
            return parse_config(yaml.safe_load(config_file))

    def validate_args(self) -> None:
        """Validate arguments shared by every runner."""
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("start_time and end_time must be provided together")
        if self.requested_num_updates is not None and self.requested_num_updates < 1:
            raise ValueError("num_updates must be a positive integer")

    def load_config(self) -> None:
        """Load and parse the forcing configuration."""
        if self.config is None:
            self.config = self._read_config(self.config_path)

    def validate_config(self) -> None:
        """Validate configuration shared by every runner."""
        if self.config["GRID_TYPE"] not in BMIMODEL:
            raise ValueError(f"Unsupported GRID_TYPE: {self.config['GRID_TYPE']}")

    @abstractmethod
    def _create_model(self):
        """Construct the BMI model for this runner type."""
        raise NotImplementedError

    @abstractmethod
    def _get_update_plan(self) -> int:
        """Return the number of model updates to run."""
        raise NotImplementedError

    def initialize_model(self) -> None:
        """Construct and initialize the BMI model."""
        print("Creating an instance of the BMI model object")
        self.model = self._create_model()
        self.model.initialize(str(self.config_path))
        self.num_updates = self._get_update_plan()
        print_init(self.model, self.num_updates)

    def _update_model(self, num_updates: int) -> None:
        """Run model updates with standard progress output when available."""
        for num in range(num_updates):
            if self.is_general:
                print_pre_update(num, num_updates, self.ngen_datetimes[num])
            self.model.update()
            if self.is_general:
                print_post_update(self.model, self.start_time)

    def _run_updates(self) -> None:
        """Run the update plan supplied by the concrete runner."""
        self._update_model(self.num_updates)

    def run_model(self) -> None:
        """Initialize, run, and finalize the BMI model."""
        try:
            self.initialize_model()
            self._run_updates()
        finally:
            if self.model is not None:
                print("\nFinalizing the BMI model")
                self.model.finalize()


class ForcingRunnerGeneral(ForcingRunner):
    """Run a hydrofabric or unstructured forcing configuration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_general = True

    def validate_args(self) -> None:
        """Validate and normalize the standard start and end times."""
        super().validate_args()
        if self.cycle_datetime is not None:
            raise ValueError(
                "cycle_datetime is only supported for gridded configurations"
            )
        if self.start_time is None or self.end_time is None:
            raise ValueError("start_time and end_time are required")
        self.ngen_datetimes, self.start_time, self.end_time = get_date_times(
            self.start_time, self.end_time
        )
        self.num_updates = len(self.ngen_datetimes)

    def validate_config(self) -> None:
        """Validate a non-gridded forcing configuration."""
        super().validate_config()
        if self.config["GRID_TYPE"] == "gridded":
            raise ValueError("ForcingRunnerGeneral does not support GRID_TYPE: gridded")
        if self.output_t0 is not False:
            raise ValueError("output_t0 is only supported for gridded configurations")

    def _create_model(self):
        """Construct the configured BMI model."""
        return BMIMODEL[self.config["GRID_TYPE"]](
            self.b_date,
            self.geogrid,
            output_path=str(self.output_path) if self.output_path else None,
        )

    def _get_update_plan(self) -> int:
        """Run one update for every requested timestamp."""
        return self.num_updates


class ForcingRunnerGridded(ForcingRunner):
    """Run a gridded forcing cycle or explicit retrospective window."""

    @property
    def explicit_window(self) -> bool:
        """Whether the runner was given a fixed start/end time window."""
        return self.start_time is not None or self.end_time is not None

    def validate_args(self) -> None:
        """Validate and normalize timing inputs."""
        super().validate_args()
        self.validate_gridded_args()

    def validate_gridded_args(self) -> None:
        """Validate and normalize gridded timing inputs."""
        if self.cycle_datetime is not None and self.explicit_window:
            raise ValueError(
                "cycle_datetime cannot be combined with start_time/end_time"
            )
        if self.requested_num_updates is not None and self.cycle_datetime is None:
            raise ValueError("num_updates requires cycle_datetime")
        if self.cycle_datetime is None and (
            self.start_time is None or self.end_time is None
        ):
            raise ValueError("Provide cycle_datetime or both start_time and end_time")
        if isinstance(self.start_time, str) and isinstance(self.end_time, str):
            _, self.start_time, self.end_time = get_date_times(
                self.start_time, self.end_time
            )
        if self.explicit_window and self.end_time < self.start_time:
            raise ValueError("end_time must not be earlier than start_time")

    def validate_config(self) -> None:
        """Load configuration and derive the gridded output count."""
        super().validate_config()
        self.validate_gridded_config()

    def validate_gridded_config(self) -> None:
        """Validate gridded configuration and derive its output count."""
        if self.config["GRID_TYPE"] != "gridded":
            raise ValueError("ForcingRunnerGridded requires GRID_TYPE: gridded")
        if self.output_t0 and self.config["AnAFlag"]:
            raise ValueError("output_t0 is not supported for analysis configurations")
        if self.explicit_window and self.config["AnAFlag"]:
            raise ValueError(
                "Explicit windows are not supported for analysis configurations; "
                "provide cycle_datetime so LookBack controls the window"
            )
        if self.explicit_window:
            duration_seconds = (self.end_time - self.start_time).total_seconds()
            step_seconds = self.config["time_step_seconds"]
            if duration_seconds % step_seconds:
                raise ValueError(
                    "The explicit window must be evenly divisible by time_step_seconds"
                )
            self.output_steps = int(duration_seconds / step_seconds) + 1

    def _create_model(self):
        """Construct the gridded BMI model with timing overrides."""
        reference_time = self.cycle_datetime or self.start_time

        # Do some math in case the user asked for the T0 output to be included.
        forecast_output_steps = self.output_steps
        if self.output_t0 and forecast_output_steps is not None:
            forecast_output_steps -= 1
        elif (
            self.output_t0
            and self.requested_num_updates is not None
            and not self.config["AnAFlag"]
        ):
            forecast_output_steps = self.requested_num_updates

        return BMIMODEL["gridded"](
            b_date=self.b_date or reference_time.strftime(REFERENCE_TIME_FORMAT),
            geogrid=self.geogrid,
            output_path=str(self.output_path) if self.output_path else None,
            output_steps=forecast_output_steps,
            output_t0=self.output_t0,
        )

    def _get_update_plan(self) -> int:
        """Use the configured or requested update count."""
        if self.explicit_window:
            return self.output_steps - int(self.output_t0)
        if self.requested_num_updates is not None:
            return self.requested_num_updates
        return self.model._job_meta.actual_output_steps - int(self.output_t0)


def run_bmi(
    start_time: str | None = None,
    end_time: str | None = None,
    config_path: pathlib.Path = None,
    b_date: str = None,
    geogrid: str = None,
    output_path: pathlib.Path = None,
    cycle_datetime: datetime.datetime | None = None,
    num_updates: int | None = None,
    output_t0: bool = False,
):
    """Execute the NextGen Forcings Engine BMI model.

    Wrapper script to execute the forcing engine BMI model. Requires user to pass start_time and end_time as arguments.
    Additionally, configurations are parsed from config.yml. Users can provide a custom config file with config_path.

    :param start_time: The start time for the simulation, in the format 'YYYY-MM-DD HH:mm:ss'.
    :param end_time: The end time for the simulation, in the format 'YYYY-MM-DD HH:mm:ss'.
    :param config_path: Optional path to the configuration file. Defaults to the config.yml bundled with this module.
    :param b_date: The begin date for the forecast cycle, in the format 'YYYYMMDDHHmm'. If omitted, reads from config.
    :param geogrid: Path to the geospatial grid file. If omitted, reads from the config file.
    :param output_path: Path to the output file. If omitted, a default output path is generated.
    :param cycle_datetime: Optional gridded forecast cycle time. Cannot be combined with start_time or end_time.
    :param num_updates: Optional positive update count for a gridded cycle. This limits iteration count without changing an analysis lookback window.
    :param output_t0:
        Write an additional T0 record before the normal gridded forecast outputs, with identical forcing values as T1.
        Not supported for analysis configurations.
        Only supported for ``gridded`` configurations.
        An independent T0 state cannot be reliably computed because several forecast products omit required fields at hour zero and substitute hour one. Examples:
        https://github.com/NGWPC/ngen-forcing/blob/27e03ba138478dd449ce957b1c3ba4c36fc33d8f/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/time_handling.py#L1202-L1206
        https://github.com/NGWPC/ngen-forcing/blob/27e03ba138478dd449ce957b1c3ba4c36fc33d8f/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/time_handling.py#L1444-L1448
        https://github.com/NGWPC/ngen-forcing/blob/27e03ba138478dd449ce957b1c3ba4c36fc33d8f/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/time_handling.py#L2043-L2047
        https://github.com/NGWPC/ngen-forcing/blob/27e03ba138478dd449ce957b1c3ba4c36fc33d8f/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/time_handling.py#L4129-L4135

    :raises RuntimeError: If the model fails to initialize or if required arguments are missing.
    """
    print("Initializing the BMI model")
    runner = ForcingRunner.create(
        config_path,
        cycle_datetime=cycle_datetime,
        start_time=start_time,
        end_time=end_time,
        b_date=b_date,
        geogrid=geogrid,
        output_path=output_path,
        num_updates=num_updates,
        output_t0=output_t0,
    )
    runner.validate_args()
    runner.load_config()
    runner.validate_config()
    runner.run_model()


def main():
    """Parse arguments and run the BMI model.

    Calls the `run_bmi` function with parsed command-line arguments.
    """
    args = get_options()
    print("run_bmi_model args:", vars(args))
    run_bmi(
        start_time=args.start_time,
        end_time=args.end_time,
        config_path=args.config_path,
        b_date=args.b_date,
        geogrid=args.geogrid,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
