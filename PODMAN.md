# Podman Support (Additive Path)

This repository maintains Docker as its primary documented and production CI path. Podman is supported as an **additional** option for local development, rootless execution, and container build/smoke verification.

Existing Docker workflows (`.github/workflows/cicd.yml`, `.github/workflows/dependencies.yml`) and production release tags remain untouched and active.

---

## Container Architecture: Two-Tier Build

`ngen-forcing` maintains two distinct Dockerfiles to optimize developer feedback loops:

1. **`Dockerfile.dependencies` (Base Image — ~45–60 min build):**
   * Compiles large, slow-moving C, C++, and Fortran dependencies from source: **Boost 1.86.0**, **ESMF 8.8.0 + ESMPy**, **ecFlow 5.15.2**, and **WGRIB2**, along with system NetCDF, HDF5, and GDAL packages.
   * This base image rarely changes and is published as `ghcr.io/ngwpc/ngen-dependencies-bookworm:latest`.

2. **`Dockerfile.bmi-forcings` (Application Image — ~2 min build):**
   * Inherits directly `FROM` the compiled dependencies image.
   * Installs the `NextGen_Forcings_Engine_BMI` Python package and EWTS.
   * This is what developers build and test during day-to-day work.

---

## Prerequisites

Verify that Podman is installed on your workstation:

```bash
podman version
podman info
```

For Ubuntu 24.04+ (Noble) or RHEL 8/9, Podman 4.9+ is recommended.

---

## Registry Authentication

When using the published base image, `Dockerfile.bmi-forcings` pulls `ghcr.io/ngwpc/ngen-dependencies-bookworm:latest`. Ensure you are logged into GHCR:

```bash
echo "<GITHUB_PAT_OR_TOKEN>" | podman login ghcr.io -u "<GITHUB_USERNAME>" --password-stdin
```

---

## Building with Podman

### 1. Building the Application Image (`Dockerfile.bmi-forcings`) — *Standard Fast Path*

This is the standard command for daily development:

```bash
podman build \
  --ulimit nofile=65535:65535 \
  --format docker \
  -f Dockerfile.bmi-forcings \
  -t local/ngen-bmi-forcing:podman-test \
  .
```

### 2. Building the Dependencies Base (`Dockerfile.dependencies`) — *Infrequent*

When updating Boost, ESMF, or system libraries, build the base image locally:

```bash
podman build \
  --ulimit nofile=65535:65535 \
  --format docker \
  -f Dockerfile.dependencies \
  -t local/ngen-dependencies-bookworm:podman-test \
  .
```

To build `Dockerfile.bmi-forcings` on top of your freshly built local base image:

```bash
podman build \
  --ulimit nofile=65535:65535 \
  --build-arg DEPS_IMAGE=local/ngen-dependencies-bookworm:podman-test \
  --format docker \
  -f Dockerfile.bmi-forcings \
  -t local/ngen-bmi-forcing:podman-test \
  .
```

*Note: The Dockerfiles use BuildKit syntax (`--mount=type=cache`) for apt and pip caching. Modern Podman (via Buildah $\ge$ 1.24) natively resolves cache mounts locally without requiring a Docker daemon. The `--ulimit nofile=65535:65535` flag ensures sufficient file descriptors are available during heavy compilation.*

---

## Smoke Verification

### Application Image (`Dockerfile.bmi-forcings`)

#### 1. Test Entrypoint & Bash Execution
```bash
podman run --rm local/ngen-bmi-forcing:podman-test -c "echo 'Entrypoint functional' && /bin/bash --version"
```

#### 2. Verify Python Environment & Dependencies
```bash
podman run --rm local/ngen-bmi-forcing:podman-test -c "python --version && python -c 'import NextGen_Forcings_Engine, ewts; print(\"Packages healthy:\", NextGen_Forcings_Engine.__version__)'"
```

#### 3. Verify Provenance Metadata
```bash
podman run --rm local/ngen-bmi-forcing:podman-test -c "cat /ngen-app/ngen-bmi-forcing_git_info.json && test -s /ngen-app/ngen-bmi-forcing_git_info.json"
```

### Dependencies Image (`Dockerfile.dependencies`)

```bash
# Verify wgrib2 binary
podman run --rm local/ngen-dependencies-bookworm:podman-test wgrib2 -v

# Verify ESMF/ESMPy Python module
podman run --rm local/ngen-dependencies-bookworm:podman-test python -c "import ESMF; print('ESMF Version:', ESMF.__version__)"

# Verify GDAL binary
podman run --rm local/ngen-dependencies-bookworm:podman-test gdalinfo --version

# Verify Boost header installation
podman run --rm local/ngen-dependencies-bookworm:podman-test test -d /opt/boost
```

---

## CI / Automation

* **Workflow:** `.github/workflows/podman-smoke.yml`
* **Triggers:**
  * Automated checks on pull requests modifying container or source files (`Dockerfile.bmi-forcings`, `Dockerfile.dependencies`, `pyproject.toml`, `NextGen_Forcings_Engine_BMI/**`, `Forcing_Extraction_Scripts/**`, `ESMF_Mesh_Domain_Configuration_Production/**`).
  * Manual triggers via `workflow_dispatch` (with optional `rebuild_dependencies: true`).
* **Smart Dependencies Strategy:**
  * On normal PRs (only modifying Python files / `Dockerfile.bmi-forcings`), CI takes the **fast path (~2 mins)** using the prebuilt `ghcr.io/ngwpc/ngen-dependencies-bookworm:latest` base.
  * If a PR modifies `Dockerfile.dependencies` (or `rebuild_dependencies: true` is dispatched), CI automatically executes the **dependencies build & smoke probe** first, and feeds that fresh local base directly into the BMI forcing build.
* **Runner Environment:** Pinned to `ubuntu-24.04`.
* **Registry Policy:** By default, builds remain local to the runner. When `push_images=true` is dispatched, only `:podman-test` and `:<sha>-podman-test` tags are published to GHCR. Production aliases (`:latest`, branch tags) are never touched.
