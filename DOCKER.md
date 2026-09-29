# How to use ngen-forcing Docker containers

The Dockerfiles within this project allow you to build the dependency and BMI
Forcings images used by the current NextGen/NWM v4 stack. The repository also
retains a deprecated Lumped Forcings Driver image.

### Dependencies Image: `Dockerfile.dependencies`

This is the base of `NextGen_Forcings_Engine_BMI`.

### NextGen_Forcings_Engine_BMI Image: `Dockerfile.bmi-forcings`

This is the primary BMI Forcings image.

### NextGen_Lumped_Forcings_Driver: `Dockerfile.lumped-forcings`

**WARNING: This is deprecated.**


## Requirements

To build and run these containers, you will need the following software installed and running on your system:
- Docker Engine

## Building Dependencies Image

To build the Dependencies image, execute the following command:

```shell
docker build --file Dockerfile.dependencies --tag "forcing-deps" .
```

## Building NextGen_Forcings_Engine_BMI

To build the NextGen_Forcings_Engine_BMI container, execute the following command:

**Note:** This Dockerfile fetches Git metadata from `origin`. A local Docker build
cannot use the host's SSH configuration by default. If this repository's remote
uses SSH, temporarily change it to HTTPS before building:

```shell
git remote set-url origin https://github.com/NGWPC/ngen-forcing.git
```

After the build, restore the SSH remote if desired:

```shell
git remote set-url origin git@github.com:NGWPC/ngen-forcing.git
```

```shell
docker build --file Dockerfile.bmi-forcings \
    --build-arg "DEPS_IMAGE=forcing-deps" \
    --build-arg "CI_COMMIT_REF_NAME=${CI_COMMIT_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}" \
    --tag "forcing-bmi-forcings" .
```

Then run the pytest suite and check the return code.

In this example, `mpirun -n 2` is prepended to the call so that it runs using 2 MPI ranks (two processes).  If wishing to run on one MPI rank (one process), simply remove `mpirun -n 2` from the command.

```shell
# Run the test:
docker run --rm \
    --env OMPI_ALLOW_RUN_AS_ROOT=1 \
    --env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    --volume "${PWD}:/workspaces/nwm-rte/src/ngen-forcing" \
    --workdir /workspaces/nwm-rte/src/ngen-forcing \
    "forcing-bmi-forcings" \
    -c 'python -m pip install ".[develop]" && mpirun -n 2 pytest'

# Check the return code:
echo $?
```

For additional details on the pytest and ways to run it from the `nwm-rte` Dev Container, see [tests/README.md](tests/README.md).

## Building NextGen_Lumped_Forcings_Driver (Deprecated)

The Lumped Forcings Driver image is retained from the original codebase but is
not used by the current dependency-to-BMI-to-ngen image chain.

To build the NextGen_Lumped_Forcings_Driver container, execute the following command:
```
docker build --file=Dockerfile.lumped-forcings --tag=ngen-lumped-forcing .
```

## Running NextGen_Forcings_Engine_BMI

To run the NextGen_Forcings_Engine_BMI container, execute the following command:
```
docker run --rm -it forcing-bmi-forcings
```
This will drop you to a bash prompt inside the container.

The Python virtual environment (`/ngen-app/ngen-python`) is already on the `PATH`,
so `python` runs the container's interpreter with all dependencies installed. No
activation step is needed.

All the ngen-forcing scripts are located at `/ngen-app/ngen-forcing/`.


## Running NextGen_Lumped_Forcings_Driver (Deprecated)

To run the NextGen_Lumped_Forcings_Driver container, execute the following command:
```
docker run -it ngen-lumped-forcing
```
This will drop you to a bash prompt inside the container.

The `ngen_lumped_forcings_driver` conda environment is already on the `PATH`, so
`python` runs with its dependencies installed.

All the ngen-forcing scripts are located at `/ngen-app/ngen-forcing/`.

## Troubleshooting

Troubleshooting information and procedures will be added as we further improve these containers.

## Future Improvements

- Add entrypoint scripts that make it easier to execute these scripts
- Replace specialized fork of ExactExtract python package with official release
