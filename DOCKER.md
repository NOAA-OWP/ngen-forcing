# How to use ngen-forcing Docker containers

The Dockerfiles within this project will allow you to run the scripts from the ngen-forcing repository. There are 2 Docker container images that can be built:
1. NextGen_Forcings_Engine_BMI
1. NextGen_Lumped_Forcings_Driver

## Requirements

To build and run these containers, you will need the following software installed and running on your system:
- Docker Engine

## Building NextGen_Forcings_Engine_BMI

To build the NextGen_Forcings_Engine_BMI container, execute the following command:
```
docker build --file=Dockerfile.bmi-forcings --tag=ngen-bmi-forcing .
```

## Building NextGen_Lumped_Forcings_Driver

To build the NextGen_Lumped_Forcings_Driver container, execute the following command:
```
docker build --file=Dockerfile.lumped-forcings --tag=ngen-lumped-forcing .
```

## Running NextGen_Forcings_Engine_BMI

To run the NextGen_Forcings_Engine_BMI container, execute the following command:
```
docker run -it ngen-bmi-forcing
```
This will drop you to a bash prompt inside the container.

You will next need to activate the required conda environment:
```
conda activate /ngen-app/conda
```
All the ngen-forcing scripts are located at `/ngen-app/ngen-forcing/`.


## Running NextGen_Lumped_Forcings_Driver

To run the NextGen_Lumped_Forcings_Driver container, execute the following command:
```
docker run -it ngen-lumped-forcing
```
This will drop you to a bash prompt inside the container.

You will next need to activate the required conda environment:
```
conda activate /ngen-app/conda
```
All the ngen-forcing scripts are located at `/ngen-app/ngen-forcing/`.

## Troubleshooting

Troubleshooting information and procedures will be added as we further improve these containers.

## Future Improvements 

- Make sure conda environments are activating automatically and don't have to be activated as a separate step
- Add entrypoint scripts that make it easier to execute these scripts
- Replace specialized fork of ExactExtract python package with official release
