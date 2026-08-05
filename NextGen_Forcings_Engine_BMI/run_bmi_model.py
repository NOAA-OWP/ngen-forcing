import argparse
import datetime
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# This is the NextGen Forcings Engine BMI instance to execute
from NextGen_Forcings_Engine.bmi_model import (
    BMIMODEL,
    NWMv3_Forcing_Engine_BMI_model,
    parse_config,
)


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
    print(f"Iteration #{num} of {num_iterations} for {timestamp}")


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


def run_bmi(
    start_time: str,
    end_time: str,
    config_path: pathlib.Path = None,
    b_date: str = None,
    geogrid: str = None,
    output_path: pathlib.Path = None,
):
    """Execute the NextGen Forcings Engine BMI model.

    Wrapper script to execute the forcing engine BMI model. Requires user to pass start_time and end_time as arguments.
    Additionally, configurations are parsed from config.yml. Users can provide a custom config file with config_path.

    :param start_time: The start time for the simulation, in the format 'YYYY-MM-DD HH:mm:ss'.
    :param end_time: The end time for the simulation, in the format 'YYYY-MM-DD HH:mm:ss'.
    :param config_path: Optional path to the configuration file. Defaults to './config.yml' if not provided.
    :param b_date: The begin date for the forecast cycle, in the format 'YYYYMMDDHHmm'. If omitted, reads from config.
    :param geogrid: Path to the geospatial grid file. If omitted, reads from the config file.
    :param output_path: Path to the output file. If omitted, a default output path is generated.

    :raises RuntimeError: If the model fails to initialize or if required arguments are missing.
    """
    print("Initializing the BMI model")
    # Set the path for the config file, using the default if none is provided
    cfg_path = (
        str(config_path)
        if config_path is not None
        else str(Path(__file__).parent.resolve() / "config.yml")
    )
    with open(cfg_path, "r") as fp:
        config = parse_config(yaml.safe_load(fp))

    print("Creating an instance of the BMI model object")
    model = BMIMODEL[config.get("GRID_TYPE")]()

    # IMPORTANT: We are not calling initialize() directly here.
    # Instead, we call initialize_with_params(), which handles
    # the initialization process and internally calls initialize().
    model.initialize_with_params(
        cfg_path,
        b_date=b_date,
        geogrid=geogrid,
        output_path=str(output_path) if output_path else None,
    )
    ngen_datetimes, start_time, end_time = get_date_times(start_time, end_time)
    num_iterations = len(ngen_datetimes)
    print_init(model, num_iterations)

    for num, timestamp in enumerate(ngen_datetimes):
        print_pre_update(num, num_iterations, timestamp)
        model.update()
        print_post_update(model, start_time)

    print("\nFinalizing the BMI model")
    model.finalize()


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
