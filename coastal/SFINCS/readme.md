# Coastal Forcing Workflow

 This project provides a modular workflow for preparing forcing data for coastal models from multiple data sources.
 It uses a configuration file to define the model, time range, and datasets to download and process.

 Authors: Lauren Schambach (Deltares USA), Mohammed Karim (RTX), Zhengtao Cui (RTX)

## Project Structure

 ```bash
 coastalforcing/
│ config.yaml # Main run configuration
| main.py # Workflow Script
│
├───domain_lists/
│ ├───sfincs/ # Domain info for SFINCS
│ └───schism/ # Domain info for SCHISM
│
├───download_data/ # Data download scripts/classes
└───process_data/ # Data processing scripts/classes

coastalmodels/
└───sfincs/
  └───texas_example/ # base model files, copied to run directory with minor run-specific changes
    sfincs.bnd
    sfincs.inp
    sfincs.nc
    sfincs.obs
    sfincs.sbg
    sfincs_nwm.src
    sfincs_ngen.src
```

### Data Download Folder > externally mounted. Define path in coastalforcing/config.yaml.

```bash
data/
└───raw/ # directory where raw
```

### Simulation Run Directory > externally mounted. Define path in coastalforcing/config.yaml.

```bash
run/
└───{coastal_model}_{start_time}/ # model run files
```

## Python Environment Setup

Python environment for this workflow:

```bash
conda create -n ngen_coastalforcing python=3.11
conda activate ngen_coastalforcing
pip install -r requirements.txt
```
