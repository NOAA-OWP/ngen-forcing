# NGWPC Forcing Engine: Refactoring, Restructuring, and Codebase Improvement Summary

**January 2026 to August 2026**

## Overview

This document summarizes the refactoring and code improvement efforts applied to the Python modules of the NextGen forcing engine (`ngen-forcing` repository) from January 2026 to August 2026. These efforts are distinct from new capabilities and feature additions. The work has improved the readability, maintainability, and extensibility of the codebase.

The improvements fall into five categories:

1. **Detailed Golden File Tests & Python Debugger Configurations**
2. **Formatting, Linting, Type-Hinting, and Style**
3. **Code Restructuring**
4. **Removing Deprecated / Unused Code and Data**
5. **Bug Discovery and Resolution**

These refactorings and code improvements are not exhaustive — some files have received more attention than others. However, the new patterns implemented can be continued.

### Primary List of Affected Files

Not every file was refactored. Here is a list of files most affected:

- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/bmi_model.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/bmi_model.py)
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/config.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/config.py)
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/forecastMod.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/forecastMod.py)
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/layeringMod.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/layeringMod.py)
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/parallel.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/parallel.py)
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/regrid.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/regrid.py) -- `regrid.py` was not extensively refactored, but initial headway was made and patterns were implemented which could be continued. Significant amounts of duplicated logic remain, more DRYification efforts are needed.
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/suppPrecipMod.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/suppPrecipMod.py)
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py)
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/model.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/model.py)
- [`NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/historical_forcing.py`](https://github.com/NGWPC/ngen-forcing/blob/NGWPC-7625_PI_10_ngen_forcing_refactor/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/historical_forcing.py)

### Pull Requests Related to Tests, Refactoring, and Repo Cleanup

| PR | Date | Title | Primary Files | Refactor Gist |
|----|------|-------|---------------|------|
| [#55](https://github.com/NGWPC/ngen-forcing/pull/55) | 2026-01-09 | Regridding Weights: unique cache file names, refactor | regrid.py, parallel.py | Refactored `calculate_weights` into smaller functions |
| [#65](https://github.com/NGWPC/ngen-forcing/pull/65) | 2026-01-29 | Remove the data and binary files from the coastal code | *(Repo Cleanup)* | Removed coastal binary and data files that had been moved to the nwm-coastal repository. |
| [#91](https://github.com/NGWPC/ngen-forcing/pull/91) | 2026-02-11 | Formatting | *(Multiple)* | Applied Ruff auto-formatting across the codebase. |
| [#98](https://github.com/NGWPC/ngen-forcing/pull/98) | 2026-02-13 | regrid.py Further Formatting and Direct Log Calls | regrid.py | Additional formatting and improved `err_handler` log calls. |
| [#99](https://github.com/NGWPC/ngen-forcing/pull/99) | 2026-02-13 | regrid.py: Partials | regrid.py | Introduced the `Partials` class to DRYify repeated log and error-handling call patterns. |
| [#100](https://github.com/NGWPC/ngen-forcing/pull/100) | 2026-02-13 | Add new OS utils for DRY file handling | os_utils.py, regrid.py | Created shared `os_utils.py` module for DRY file operations (remove, close, symlink). |
| [#113](https://github.com/NGWPC/ngen-forcing/pull/113) | 2026-03-10 | Test for regrid.py | regrid.py, tests | Added golden file tests for regrid operations. |
| [#125](https://github.com/NGWPC/ngen-forcing/pull/125) | 2026-04-07 | Tests for GeoMeta and InputForcings | tests | Added golden file tests for GeoMeta prior to refactoring. |
| [#105](https://github.com/NGWPC/ngen-forcing/pull/105) | 2026-04-08 | GeoMeta Linting | geoMod.py | Applied linting and formatting to `geoMod.py`. |
| [#126](https://github.com/NGWPC/ngen-forcing/pull/126) | 2026-04-08 | Modularize GeoMeta | geoMod.py | Broke out monolithic functions into smaller modular units. |
| [#127](https://github.com/NGWPC/ngen-forcing/pull/127) | 2026-04-08 | Refactor InputForcings (Part 1) | forcingInputMod.py, geoMod.py | Refactored `forcingInputMod.py` with OOP inheritance for discretization types. |
| [#128](https://github.com/NGWPC/ngen-forcing/pull/128) | 2026-04-08 | Refactor InputForcings (Part 2) | forcingInputMod.py | Continued OOP refactor of `forcingInputMod.py` with extracted constants. |
| [#101](https://github.com/NGWPC/ngen-forcing/pull/101) | 2026-04-08 | Refactor GeoMetaWrfHydro | geoMod.py | Restructured WRF/geo/hydro metadata handling with OOP inheritance. |
| [#137](https://github.com/NGWPC/ngen-forcing/pull/137) | 2026-04-16 | Refactor GeoMeta, InputForcings, NWMv3_Forcing_Engine_BMI_model, + Tests | geoMod.py, forcingInputMod.py, bmi_model.py | Further refactoring of GeoMeta and InputForcings with accompanying tests. Also establish OOP class inheritance structure for bmi_model.py. |
| [#145](https://github.com/NGWPC/ngen-forcing/pull/145) | 2026-04-25 | Repo cleanup | *(repo cleanup)* | General repository cleanup of unused files and artifacts. |
| [#147](https://github.com/NGWPC/ngen-forcing/pull/147) | 2026-05-06 | Added bmi_model tests | *(tests)* | Added golden file tests for `BmiModel` class attributes. |
| [#148](https://github.com/NGWPC/ngen-forcing/pull/148) | 2026-05-08 | Add Test for CONUS Standard AnA and Refactor Test Configuration | *(tests)* | Added AnA test configuration and refactored test framework for extensibility. |
| [#196](https://github.com/NGWPC/ngen-forcing/pull/196) | 2026-07-13 | Remove streamflow scripts | *(repo cleanup)* | Removed streamflow scripts that were moved to `nwm-data-assimilation`. |
| [#197](https://github.com/NGWPC/ngen-forcing/pull/197) | 2026-07-14 | Run flynt on downscale.py | downscale.py | Converted string formatting to f-strings via flynt. |
| [#198](https://github.com/NGWPC/ngen-forcing/pull/198) | 2026-07-14 | Run flynt on bias_correction.py | bias_correction.py | Converted string formatting to f-strings via flynt. |
| [#199](https://github.com/NGWPC/ngen-forcing/pull/199) | 2026-07-14 | Run flynt on forecast_mod.py | bias_correction.py, forecastMod.py | Converted string formatting to f-strings via flynt. |
| [#200](https://github.com/NGWPC/ngen-forcing/pull/200) | 2026-07-14 | Add basic type hints to downscale.py | downscale.py | Added type hints and renamed args for PEP 8 consistency. |
| [#207](https://github.com/NGWPC/ngen-forcing/pull/207) | 2026-07-30 | Update Test Data for NHF 1.2.2 and Latest Forcing Class Structures | *(tests)* | Updated golden file test data for NHF 1.2.2 hydrofabric and latest class structures. |
| [#134](https://github.com/NGWPC/ngen-forcing/pull/134) | 2026-08-13 | Refactor of Supplemental Precipitation Mod (suppPrecipMod.py) | suppPrecipMod.py, consts.py | Refactored into OOP inheritance with parent/child classes per discretization type. |
| [#201](https://github.com/NGWPC/ngen-forcing/pull/201) | 2026-08-13 | Refactor parallel.py (NGWPC-10583) | parallel.py | Simplified MPI methods, added docs, replaced `atexit` with explicit cleanup. |
| [#191](https://github.com/NGWPC/ngen-forcing/pull/191) | 2026-08-26 | Refactor: bmi_model.py, config.py, model.py | bmi_model.py, config.py, model.py | bmi_model.py: decomposed initialization into focused setup methods with property access. config.py: split config parsing into property setters with validation. model.py: broke run loop into named steps and extracted dispatch logic. |
| [#224](https://github.com/NGWPC/ngen-forcing/pull/224) | 2026-08-26 | Refactor layeringMod.py | layeringMod.py | Established abstract parent class with discretization-specific children to replaced if/elif blocks. |
| [#226](https://github.com/NGWPC/ngen-forcing/pull/226) | 2026-08-26 | Refactor timeInterpMod.py | timeInterpMod.py | Reorganized top-level functions into a `_TimeInterp` class with smaller methods. |

# Summary of Improvements by Category

## 1. Detailed Golden File Tests & Python Debugger Configurations

The tests originally inherited in the `ngen-forcing` repository asserted that the BMI interface methods would run without throwing an exception, but did not provide a mechanism for inspecting the results of the processes nor for confirming that changes to the code were not causing unexpected changes in the outputs.

The new golden file tests added in 2026 perform a serialization (to disk) of various class attribute structures. The serialization operations occur at initialization, at timesteps 0, 1, and 2, and at finalization. This is a minimal amount of timesteps needed to reach certain parts of the flow.

These dumped JSON files are committed to the repository as "expected" data -- commonly called "golden files". With these files in place. When the tests run in a normal fashion, they serialize the equivalent "actual" data structures to JSON on disk, and then confirm that the two are equivalent.

Some numerical tolerance is accounted for, and there are convenient ways to exclude certain keys from the set of keys that are dumped and checked. For structures that are too large, they are replaced by a string that contains metadata: the length of the object and a hash of its values.

The tests are structured with care for convenient and idiomatic extensibility going forward, to increase coverage as needed.

### Testing Environment

**NOTE**: The example commands listed in [tests/README.md](tests/README.md) assume the user is running in the Dev Container environment provided by the `nwm-rte` repository.

### Coverage and Limitations of Tests

Timesteps 0, 1, and 2 are covered.

The forcing configurations covered include default configurations for Short Range and Analysis & Assimilation realizations, as well as AORC usage for historical realizations.

The tests currently only cover the **Hydrofabric** discretization type (not the **Gridded** or **Unstructured** discretization types).

While the current golden file tests do not provide 100% coverage of every manner in which the forcing engine can be run, they provide a reliable baseline for confirming that code changes produce numerically identical results, with configurable tolerance (absolute and relative).

### Python Debugger

Python debugger configurations were added which allow developers to execute `run_bmi_model.py` with `debugpy`, with either one MPI process or 2 MPI processes.

The `-n 2` (2 MPI processes) configuration is particularly important for being able to supervise the call stack and variables' states of each MPI rank in real time. Placing a breakpoint at a particular line in the debugger causes both ranks to pause at the same location. Without this capability, it can be particularly difficult to debug MPI-related nuances, especially considering how MPI rank 0 naturally has different logic paths than other MPI ranks.

## 2. Formatting, Linting, Type-Hinting, and Style

Automatic formatting was applied to improve consistency and readability of the code. **Ruff** was the primary tool applied for automatic formatting. **Flynt** was also used to replace existing antipattern string composition approaches with modern Python f-strings.

Variables were renamed to follow general PEP 8 guidelines on case.

Type hints were added. Some circular imports were avoided by placing type-only imports behind `if TYPE_CHECKING:` guards, which prevent the imports from executing at runtime while still allowing static type checkers and IDEs to resolve the references.

## 3. Code Restructuring

### 3.1 Long Monolithic Functions Broken into Smaller Units

Long monolithic functions were decomposed into smaller units with improved (reduced) scope.

**Example — `config.py`:** The original `read_config()` method was spanned over 1000 lines. In the refactored code, configuration parsing is handled by individual property setters, each responsible for validating and storing a single configuration attribute.
  - [Before](https://github.com/NGWPC/ngen-forcing/blob/02e47e64555b7704e9f00c03bdac07fbb127ca2c/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/config.py#L18) · [After](https://github.com/NGWPC/ngen-forcing/blob/2bddeeaa59f49ee66c7c59967c5b8d2f5f16bb45/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/config.py#L31)

**Example — `timeInterpMod.py`:** The original module had long top-level functions, each containing `if` / `elif` / `else` conditional branches to handle the 3 discretization types, of which the majority of the business logic was replicated among those 3. The refactored module organizes these into a `_TimeInterp` class with shared business logic and smaller methods.
  - [Before](https://github.com/NGWPC/ngen-forcing/blob/02e47e64555b7704e9f00c03bdac07fbb127ca2c/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L9) · [After](https://github.com/NGWPC/ngen-forcing/blob/2bddeeaa59f49ee66c7c59967c5b8d2f5f16bb45/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L46)

### 3.2 OOP Inheritance for Discretization Types

Long `if`/`elif`/`elif` conditional flows were replaced with OOP class inheritance designs. The 3 main discretization modes — **Hydrofabric**, **Gridded**, and **Unstructured** — have been separated into individual classes that inherit from a common parent class. The abstract `_LayeringMod` parent class has three concrete children: `_LayeringMod_Gridded`, `_LayeringMod_Unstructured`, and `_LayeringMod_Hydrofabric`.

Examples: `suppPrecipMod.py` and `forcingInputMod.py`.

  - `layeringMod.py`: [Before](https://github.com/NGWPC/ngen-forcing/blob/02e47e64555b7704e9f00c03bdac07fbb127ca2c/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/layeringMod.py#L8) · [After](https://github.com/NGWPC/ngen-forcing/blob/2bddeeaa59f49ee66c7c59967c5b8d2f5f16bb45/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/layeringMod.py#L38)
  - `suppPrecipMod.py`: [Before](https://github.com/NGWPC/ngen-forcing/blob/02e47e64555b7704e9f00c03bdac07fbb127ca2c/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/suppPrecipMod.py#L12) · [After](https://github.com/NGWPC/ngen-forcing/blob/2bddeeaa59f49ee66c7c59967c5b8d2f5f16bb45/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/suppPrecipMod.py#L28)

### 3.3 Extraction of Constants to `consts.py`

Configuration values for the various forcing input data sources, configuration modes, and magic numbers were moved into a new shared file, `consts.py`. This centralizes these large objects that previously were scattered across multiple files, mixed-in with business logic.

### 3.4 Shared Utilities

Duplicated logic was replaced with shared utilities, for example the new file [NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/os_utils.py](https://github.com/NGWPC/ngen-forcing/blob/2bddeeaa59f49ee66c7c59967c5b8d2f5f16bb45/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/os_utils.py#L23) for DRY file handling.

### 3.5 `regrid.py`: Improvements and Remaining Work

`regrid.py` is one of the largest files in the codebase. It was not fully refactored between January and August 2026, but it did receive some improvements. Significant refactoring is still needed to reduce duplicated logic and convert the flow into a more human-readable state.

Here are some of the improvements that were made:

- **Auto-formatting** via Ruff and flynt for consistent style and modern f-strings.
- **Logic refactoring in `calculate_weights`**: The regridding steps at the bottom of `calculate_weights` were refactored into smaller functions and common logic among branching flow paths was consolidated.
- **DRY OS operations and log calls:** Shared logic is now used for OS operations (file creation, removal, symbolic links) via `os_utils`, replacing scattered inline implementations.
- **`Partials` class:** A class of `functools.partial` objects was defined to share logic for the types of log and error-handling calls that go through the `err_handler` module. These calls affect program state and can cause intentional exits, so centralizing them reduces the risk of inconsistent error handling across the many regridding functions, and usage of the partials causes a significant reduction in total lines of code across the file.
  - [Example](https://github.com/NGWPC/ngen-forcing/blob/2bddeeaa59f49ee66c7c59967c5b8d2f5f16bb45/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/regrid.py#L85)

### 3.6 Setters, Getters, and Properties

In some cases, explicit property setters/getters were defined. These allow the implementation code to be more readable by colocating data validation logic with the definition of the attribute. In some cases these were leveraged to make properties read-only after initial set, or "hardened", for safety.

## 4. Removal of Deprecated Code and Data

- Coastal-related files were moved to the dedicated `nwm-coastal` repository.
- Streamflow scripts were moved to the `nwm-data-assimilation` repository.
- `forecastMod.py` was deprecated. This file appears to be unused by the current ngen stack; it now raises a `NotImplementedError` if another codebase attempts to import it or run it.

## 5. Bug Discovery and Resolution

Some inherited bugs and potential bugs (requiring further investigation) were discovered during the refactoring work. Some were resolved, while others were tagged with TODO comments.

Example: `perform_downscaling` used `[1] in list` instead of `1 in list`:
  - [Before](https://github.com/NGWPC/ngen-forcing/blob/02e47e64555b7704e9f00c03bdac07fbb127ca2c/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/config.py#L921) · [After](https://github.com/NGWPC/ngen-forcing/blob/2bddeeaa59f49ee66c7c59967c5b8d2f5f16bb45/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/config.py#L1580)
