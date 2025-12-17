#!/bin/bash

# Define valid commands
VALID_COMMANDS=("forecast_forcing")

# This shell script lives in the ngen-forcing repo.  It is used by CerfServer when calling ngen-forcing
#
# It is used by CerfServer directly when running in LOCAL mode.
# It is used by the ngen-forcing docker container when the server is running in DOCKER or PARALLEL_WORKS mode.

FORECAST_FORCING_DOWNLOAD_SCRIPT=/ngen-app/ngen-forcing/NextGen_Forcings_Engine_BMI/bmi_wrapper.py

# Set the umask so files and directories are created with 777 permissions
umask 000

# Function to display help message
show_help() {
  echo "Usage: $(basename "$0") <command> <cycle_name> <gpkg_file> <config_file> <forcing_dir> [stdout_file] [conda_env]"
  echo ""
  echo ""
  echo "COMMAND:"
  echo "  forecast_forcing          Run forcing download script."
  echo ""
  echo "CYCLE_NAME: Cycle name to use."
  echo "GPKG_FILE: Filename of the GPKG file."
  echo "CONFIG_FILE: Path to the wrapper config file."
  echo "FORCING_DIR: Path to the forcing directory to be populated."
  echo "STDOUT_FILE (optional): Path to the stdout file where the script's console output will be saved.  Used when running in LOCAL or DOCKER environment"
  echo "CONDA_ENV (optional): Name of the CONDA environment  Used when running in the LOCAL environment."
  echo ""
  echo "Examples:"
  echo "  $(basename "$0") forecast_forcing short_range 01123000.gpkg wrapper_config.yml test_data/forcing.nc"
  echo "  $(basename "$0") forecast_forcing short_range 01123000.gpkg wrapper_config.yml test_data/forcing.nc /path/to/output/ngen-forcing.log conda_env"
  echo ""
  exit 1
}

# Show help if the user requests it with --help or -h
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  show_help
fi

# Check if the command for the script is provided as the first argument
if [ -z "$1" ]; then
  echo "Error: No script command provided. Allowable commands are: ${VALID_COMMANDS[*]}."
  show_help
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
  echo
  echo "Error: Conda not found in the PATH. Please ensure Conda is installed and the PATH is configured correctly."
  echo
  exit 1
fi

# Get the script command and select the corresponding script path
SCRIPT_COMMAND=$1
shift 1

case "$SCRIPT_COMMAND" in
  "forecast_forcing")
    SCRIPT_PATH=$FORECAST_FORCING_DOWNLOAD_SCRIPT
    REQUIRED_ARGS=4
    ;;
  *)
    echo "Error: Invalid script command: '$SCRIPT_COMMAND'. Allowable commands are: ${VALID_COMMANDS[*]}."
    show_help
    ;;
esac

# Check if the selected script exists
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: Script not found at $SCRIPT_PATH"
  exit 1
fi

# Check if the correct number of arguments are provided for the selected command
if [ $# -lt $REQUIRED_ARGS ]; then
  echo "Error: Insufficient arguments. $SCRIPT_COMMAND requires $REQUIRED_ARGS arguments."
  show_help
fi

CYCLE_NAME=$1
GPKG_FILE=$2
CONFIG_FILE=$3
FORCING_DIR=$4
shift $REQUIRED_ARGS

echo "CYCLE_NAME: ${CYCLE_NAME}"
echo "GPKG_FILE: ${GPKG_FILE}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "FORCING_DIR: ${FORCING_DIR}"

# Check if the forcing data exists
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config file not found at ${CONFIG_FILE}"
fi

if [ $# -ge 1 ]; then
  STDOUT_FILE=$1
  echo "Output file: $STDOUT_FILE"

  # Create output directory if it doesn't exist
  STDOUT_DIR=$(dirname "$STDOUT_FILE")
  if [ ! -d "$STDOUT_DIR" ]; then
    mkdir --parents "$STDOUT_DIR"
  fi

  shift 1
fi

if [ $# -ge 1 ]; then
  CONDA_ENV=$1
  echo "Virtual environment: $CONDA_ENV"
  shift 1
fi

# Activate the virtual environment if provided
if [ -n "$CONDA_ENV" ]; then
    CONDA_CMD="conda run -n $CONDA_ENV"
else
  echo "No Conda environment provided, running with default Python environment."
  CONDA_CMD=""
fi

# Run the Python script, redirecting its output if an output file is provided
echo "   Running ${CONDA_CMD} $(basename "$SCRIPT_PATH") with: $GPKG_FILE $CYCLE_NAME $CONFIG_FILE $FORCING_DIR"
if [ -z "$STDOUT_FILE" ]; then
  $CONDA_CMD python3 "${SCRIPT_PATH}" "${CYCLE_NAME}" "${GPKG_FILE}" "-config_input" "${CONFIG_FILE}" "-csv_path" "${FORCING_DIR}"
else
  $CONDA_CMD python3 "${SCRIPT_PATH}" "${CYCLE_NAME}" "${GPKG_FILE}" "-config_input" "${CONFIG_FILE}" "-csv_path" "${FORCING_DIR}" > "${STDOUT_FILE}" 2>&1

fi

python_exit_code=$?
if [ $python_exit_code -ne 0 ]; then
  echo "$(basename "$SCRIPT_PATH") exited with code $python_exit_code"
fi

# Display output if redirected to a file
if [ -n "$STDOUT_FILE" ]; then
  echo "Output from running $(basename "$SCRIPT_PATH")"
  echo "-------------- start of $STDOUT_FILE -----------------------------"
  cat "$STDOUT_FILE"
  echo "---------------- end of $STDOUT_FILE -----------------------------"
fi

echo "Done running $(basename "$SCRIPT_PATH")"

exit $python_exit_code
