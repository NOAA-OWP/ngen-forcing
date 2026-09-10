"""Temporal interpolation input forcings to the current output timestep.

This file was refactored in August/September 2026, with the general goal of preserving existing business logic
while making it more DRY, abstracting shared logic, and aligning with PEP8 on docstrings, type hints, and symbol names.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
        ConfigOptions,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.forcingInputMod import (
        InputForcings,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import (
        MpiConfig,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.suppPrecipMod import (
        SupplementalPrecip,
    )

from . import err_handler


def arr_set_scalar(arr: np.ndarray, scalar: numbers.Real, ndim: int):
    """Use numpy slice(None) syntax to modify the array in-place by assigning any number of dimensions to the provided scalar value."""
    if not 0 <= ndim <= 99:
        raise ValueError(f"Unexpected value for ndim: {ndim}")
    dims_tup = (slice(None),) * ndim
    arr[dims_tup] = scalar


def arr_set_arr(arr_1: np.ndarray, arr_2: np.ndarray, ndim: int):
    if not 0 <= ndim <= 99:
        raise ValueError(f"Unexpected value for ndim: {ndim}")
    dims_tup = (slice(None),) * ndim
    arr_1[dims_tup] = arr_2[dims_tup]


class _TimeInterp:
    """Class for handling temporal interpolation of forcing arrays.

    Some nested attrs are reassigned to top-level class attrs for readability / noise reduction
    in the methods that use them.

    The methods of this private class contain the actual business logic of the public functions of timeInterpMod which use this class.

    For docstrings explaining the business logic and domain-level synopses, see the docstrings of the public calling functions.

    A common pattern among the methods, to keep the discretization methods DRY ("gridded", "unstructured", "hydrofabric") is to
    parameterize the dimensionality of the array reading and writing behavior and set those parameters at the top of each method.
    Also, the unstructured discretization adds "*_elem" attr interactions, so this has been parameterized into a Boolean "also_elem"
    that conditionally (when True) causes those _elem behaviors to be executed.

    This class was written during a refactoring effort in 2026. At the time of writing the refactor,
    the vast majority of the code in each method is taken verbatim from the original function, but with
    parameterization added as explained above, to keep things DRY among the discretization types.
    """

    def __init__(
        self,
        input_forcings: InputForcings,
        supplemental_precip: SupplementalPrecip,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
    ):
        self.input_forcings = input_forcings
        self.supplemental_precip = supplemental_precip
        self.config_options = config_options
        self.mpi_config = mpi_config

        self.globalNdv = self.config_options.globalNdv

        if input_forcings is not None:
            self.final_forcings = self.input_forcings.final_forcings
            self.regridded_forcings2 = self.input_forcings.regridded_forcings2
            self.regridded_forcings1 = self.input_forcings.regridded_forcings1
            # _elem equivalents
            self.final_forcings_elem = self.input_forcings.final_forcings_elem
            self.regridded_forcings2_elem = self.input_forcings.regridded_forcings2_elem
            self.regridded_forcings1_elem = self.input_forcings.regridded_forcings1_elem

        if supplemental_precip is not None:
            self.final_supp_precip = self.supplemental_precip.final_supp_precip
            self.regridded_precip2 = self.supplemental_precip.regridded_precip2
            self.regridded_precip1 = self.supplemental_precip.regridded_precip1
            # _elem equivalents
            self.final_supp_precip_elem = (
                self.supplemental_precip.final_supp_precip_elem
            )
            self.regridded_precip2_elem = (
                self.supplemental_precip.regridded_precip2_elem
            )
            self.regridded_precip1_elem = (
                self.supplemental_precip.regridded_precip1_elem
            )

    def _no_interpolation(self) -> None:
        """Perform the business logic of the *input forcings* 'no_interpolation' option.

        See the docstring of the public calling function ``no_interpolation`` for details.
        Business logic and comments from original code:
            https://github.com/NGWPC/ngen-forcing/blob/a0f217f06a0045d9f139bfa14abe711fc6f248b0/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L8-L48
        """
        ndim, also_elem = {
            "gridded": (3, False),
            "unstructured": (2, True),
            "hydrofabric": (2, False),
        }[self.config_options.grid_type]

        # Check to make sure we have valid grids.
        if self.regridded_forcings2 is None:
            arr_set_scalar(self.final_forcings, self.globalNdv, ndim)
            if also_elem:
                arr_set_scalar(self.final_forcings_elem, self.globalNdv, ndim)
        else:
            arr_set_arr(self.final_forcings, self.regridded_forcings2, ndim)
            if also_elem:
                arr_set_arr(
                    self.final_forcings_elem, self.regridded_forcings2_elem, ndim
                )

    def _no_interpolation_supp_pcp(self):
        """Perform the business logic of the *supplemental precipitation* 'no_interpolation' option.

        See the docstring of the public calling function ``no_interpolation_supp_pcp`` for details.
        Business logic and comments from original code:
            https://github.com/NGWPC/ngen-forcing/blob/a0f217f06a0045d9f139bfa14abe711fc6f248b0/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L51-L92
        """
        ndim, also_elem = {
            "gridded": (2, False),
            "unstructured": (1, True),
            "hydrofabric": (1, False),
        }[self.config_options.grid_type]

        if self.regridded_precip2 is not None:
            arr_set_arr(self.final_supp_precip, self.regridded_precip2, ndim)
            if also_elem:
                arr_set_arr(
                    self.final_supp_precip_elem, self.regridded_precip2_elem, ndim
                )
        else:
            # We have missing files.
            arr_set_scalar(self.final_supp_precip, self.globalNdv, ndim)
            if also_elem:
                arr_set_scalar(self.final_supp_precip_elem, self.globalNdv, ndim)

    def _nearest_neighbor(self):
        """Perform the business logic of the *input forcings* 'nearest_neighbor' option.

        See the docstring of the public calling function ``nearest_neighbor`` for details.
        Business logic and comments from original code:
            https://github.com/NGWPC/ngen-forcing/blob/a0f217f06a0045d9f139bfa14abe711fc6f248b0/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L95-L194
        """
        ndim, also_elem = {
            "gridded": (3, False),
            "unstructured": (2, True),
            "hydrofabric": (2, False),
        }[self.config_options.grid_type]

        # If we are running CFSv2 with bias correction, bypass as temporal interpolation is done
        # internally (NWM-only).
        if (
            self.config_options.runCfsNldasBiasCorrect
            and self.input_forcings.product_name == "CFSv2_6Hr_Global_GRIB2"
        ):
            if self.mpi_config.rank == 0:
                self.config_options.statusMsg = "Bypassing temporal interpolation routine due to NWM bias correction for CFSv2"
                err_handler.log_msg(self.config_options, self.mpi_config)
            return

        # Calculate the difference between the current output timestep,
        # and the previous input forecast output step.
        dtFromPrevious = (
            self.config_options.current_output_date - self.input_forcings.fcst_date1
        )

        # Calculate the difference between the current output timesetp,
        # and the next forecast output step.
        dtFromNext = (
            self.config_options.current_output_date - self.input_forcings.fcst_date2
        )

        if abs(dtFromNext.total_seconds()) <= abs(dtFromPrevious.total_seconds()):
            # Default to the regridded states from the next forecast output step.
            if self.regridded_forcings2 is None:
                arr_set_scalar(self.final_forcings, self.globalNdv, ndim)
                if also_elem:
                    arr_set_scalar(self.final_forcings_elem, self.globalNdv, ndim)
            else:
                arr_set_arr(self.final_forcings, self.regridded_forcings2, ndim)
                if also_elem:
                    arr_set_arr(
                        self.final_forcings_elem, self.regridded_forcings2_elem, ndim
                    )
        else:
            # Default to the regridded states from the previous forecast output
            # step.
            if self.regridded_forcings1 is None:
                arr_set_scalar(self.final_forcings, self.globalNdv, ndim)
                if also_elem:
                    arr_set_scalar(self.final_forcings_elem, self.globalNdv, ndim)
            else:
                arr_set_arr(self.final_forcings, self.regridded_forcings1, ndim)
                if also_elem:
                    arr_set_arr(
                        self.final_forcings_elem, self.regridded_forcings1_elem, ndim
                    )

    def _nearest_neighbor_supp_pcp(self):
        """Perform the business logic of the *supplemental precipitation* 'nearest_neighbor' option.

        TODO review the calculation for ``dtFromPrevious``:
        In the original business logic here, it used ``dtFromPrevious = ...current_output_step - ...pcp_date1``.
        In other functions, the original business logic used ``dtFromPrevious = ...current_output_date - ...pcp_date1``.
        Both were preserved during the refactor, but it is not clear whether this was a typo in this function, maybe it was
        intending to use ``current_output_date`` instead of ``current_output_step``?

        See the docstring of the public calling function ``nearest_neighbor_supp_pcp`` for details.
        Business logic and comments from original code:
            https://github.com/NGWPC/ngen-forcing/blob/a0f217f06a0045d9f139bfa14abe711fc6f248b0/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L197-L279
        """
        ndim, also_elem = {
            "gridded": (2, False),
            "unstructured": (1, True),
            "hydrofabric": (1, False),
        }[self.config_options.grid_type]

        if self.regridded_precip2 is None or self.regridded_precip1 is None:
            return

        # Calculate the difference between the current ouptut timestep,
        # and the previous supplemental input step.
        dtFromPrevious = (
            self.config_options.current_output_step - self.supplemental_precip.pcp_date1
        )

        # Calculate the difference between the current output timestep,
        # and the next supplemental input step.
        dtFromNext = (
            self.config_options.current_output_date - self.supplemental_precip.pcp_date2
        )

        if abs(dtFromNext.total_seconds()) <= abs(dtFromPrevious.total_seconds()):
            # Default to the regridded states from the next forecast output step.
            arr_set_arr(self.final_supp_precip, self.regridded_precip2, ndim)
            if also_elem:
                arr_set_arr(
                    self.final_supp_precip_elem, self.regridded_precip2_elem, ndim
                )
        else:
            # Default to the regridded states from the previous forecast output
            # step.
            arr_set_arr(self.final_supp_precip, self.regridded_precip1, ndim)
            if also_elem:
                arr_set_arr(
                    self.final_supp_precip_elem, self.regridded_precip1_elem, ndim
                )

    def _weighted_average(self):
        """Perform the business logic of the *input forcings* 'weighted_average' option.

        TODO Review how each ``if ... is None`` case returns early, which is from the original business logic.
        It is unclear whether this was intentional, but it was preserved in the 2026 refactor.

        NOTE in original code, for the "unstructured" case:
          ``ind1Ndv_elem`` and ``ind2Ndv_elem`` were getting assigned but never used.
          That may have been a bug, since the _elem logic path was applying the ``ind1Ndv`` mask.
          During the 2026 refactor, it was assumed that the intent was to actually use ``ind1Ndv_elem``
          and ``ind2Ndv_elem`` after setting them, so the refactored code does use them.

        See the docstring of the public calling function ``weighted_average`` for details.
        Business logic and comments from original code:
            https://github.com/NGWPC/ngen-forcing/blob/a0f217f06a0045d9f139bfa14abe711fc6f248b0/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L282-L459
        """
        ndim, also_elem = {
            "gridded": (3, False),
            "unstructured": (2, True),
            "hydrofabric": (2, False),
        }[self.config_options.grid_type]

        if self.regridded_forcings2 is None:
            arr_set_scalar(self.final_forcings, self.globalNdv, ndim)
            return
        if self.regridded_forcings1 is None:
            arr_set_scalar(self.final_forcings, self.globalNdv, ndim)
            return
        if also_elem:
            if self.regridded_forcings2_elem is None:
                arr_set_scalar(self.final_forcings_elem, self.globalNdv, ndim)
                return
            if self.regridded_forcings1_elem is None:
                arr_set_scalar(self.final_forcings_elem, self.globalNdv, ndim)
                return

        # If we are running CFSv2 with bias correction, bypass as temporal interpolation is done
        # internally (NWM-only).
        if (
            self.config_options.runCfsNldasBiasCorrect
            and self.input_forcings.product_name == "CFSv2_6Hr_Global_GRIB2"
        ):
            if self.mpi_config.rank == 0:
                self.config_options.statusMsg = "Bypassing temporal interpolation routine due to NWM bias correction for CFSv2"
                err_handler.log_msg(self.config_options, self.mpi_config)
            return

        # Calculate the difference between the current output timestep,
        # and the previous input forecast output step. Use this to calculate a fraction
        # of the previous forcing output to use in the final output for this step.
        dtFromPrevious = (
            self.config_options.current_output_date - self.input_forcings.fcst_date1
        )
        weight1 = 1 - (
            abs(dtFromPrevious.total_seconds()) / (self.input_forcings.outFreq * 60.0)
        )

        # Calculate the difference between the current output timesetp,
        # and the next forecast output step. Use this to calculate a fraction of
        # the next forcing output to use in the final output for this step.
        dtFromNext = (
            self.config_options.current_output_date - self.input_forcings.fcst_date2
        )
        weight2 = 1 - (
            abs(dtFromNext.total_seconds()) / (self.input_forcings.outFreq * 60.0)
        )

        # Calculate where we have missing data in either the previous or next forcing dataset.
        ind1Ndv = np.where(self.regridded_forcings1 == self.globalNdv)
        ind2Ndv = np.where(self.regridded_forcings2 == self.globalNdv)
        if also_elem:
            ind1Ndv_elem = np.where(self.regridded_forcings1_elem == self.globalNdv)
            ind2Ndv_elem = np.where(self.regridded_forcings2_elem == self.globalNdv)

        arr_set_arr(
            self.final_forcings,
            (
                self.regridded_forcings1[(slice(None),) * ndim] * weight1
                + self.regridded_forcings2[(slice(None),) * ndim] * weight2
            ),
            ndim,
        )
        if also_elem:
            arr_set_arr(
                self.final_forcings_elem,
                (
                    self.regridded_forcings1_elem[(slice(None),) * ndim] * weight1
                    + self.regridded_forcings2_elem[(slice(None),) * ndim] * weight2
                ),
                ndim,
            )

        # Set any pixel cells that were missing for either window to missing value.
        self.final_forcings[ind1Ndv] = self.globalNdv
        self.final_forcings[ind2Ndv] = self.globalNdv
        if also_elem:
            self.final_forcings_elem[ind1Ndv_elem] = self.globalNdv
            self.final_forcings_elem[ind2Ndv_elem] = self.globalNdv

        # Reset for memory efficiency.
        ind1Ndv = None
        ind2Ndv = None
        if also_elem:
            ind1Ndv_elem = None
            ind2Ndv_elem = None

    def __calc_weights_for_supp_pcp(self) -> tuple[float | None, float | None]:
        """Calculate weights for supplemental precip weighting.

        Return (None, None) unless both the current and previous are non-None
        (for either the non-_elem case or the _elem case).)

        For ``weight1``:
            Calculate the difference between the current output timestep,
            and the previous input supp pcp step. Use this to calculate a fraction
            of the previous supp pcp to use in the final output for this step.

        For ``weight2``:
            Calculate the difference between the current output timesetp,
            and the next input supp pcp step. Use this to calculate a fraction of
            the next forcing supp pcp to use in the final output for this step.

        Business logic and comments from original code (duplicated among the discretization types):
            https://github.com/NGWPC/ngen-forcing/blob/a0f217f06a0045d9f139bfa14abe711fc6f248b0/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L473-L497
        """

        if not (
            (self.regridded_precip2 is not None and self.regridded_precip1 is not None)
            or (
                self.regridded_precip2_elem is not None
                and self.regridded_precip1_elem is not None
            )
        ):
            return None, None

        # weight1
        dtFromPrevious = (
            self.config_options.current_output_date - self.supplemental_precip.pcp_date1
        )
        weight1 = 1 - (
            abs(dtFromPrevious.total_seconds())
            / (self.supplemental_precip.input_frequency * 60.0)
        )

        # weight2
        dtFromNext = (
            self.config_options.current_output_date - self.supplemental_precip.pcp_date2
        )
        weight2 = 1 - (
            abs(dtFromNext.total_seconds())
            / (self.supplemental_precip.input_frequency * 60.0)
        )

        return weight1, weight2

    def _weighted_average_supp_pcp(
        self, weight1: float | None, weight2: float | None, attr_suffix: str = ""
    ):
        """Perform the business logic of the *supplemental precip* 'weighted_average' option.

        ``attr_suffix`` can be empty string or "_elem". The "_elem" choices is only supported for the "unstructured" discretization.

        In the original code, for the "unstructured" discretization, the calculations of ``weight1`` and ``weight2`` were identical
        between the non-_elem and the _elem logic paths. So when this was refactored, that block was moved to a shared inner function
        ``_calc_weights``. This was decorated with lru_cache for the "unstructured" where it is called twice.

        Business logic and comments from original code:
            https://github.com/NGWPC/ngen-forcing/blob/a0f217f06a0045d9f139bfa14abe711fc6f248b0/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L462-L676
        """
        ndim = {
            "gridded": 2,
            "unstructured": 1,
            "hydrofabric": 1,
        }[self.config_options.grid_type]

        if attr_suffix == "":
            pass
        elif attr_suffix == "_elem":
            if self.config_options.grid_type != "unstructured":
                raise ValueError(
                    "attr_suffix '_elem' is only supported for the 'unstructured' grid type."
                )
        else:
            raise ValueError(f"Unexpected attr_suffix: {repr(attr_suffix)}")

        if (
            getattr(self, f"regridded_precip2{attr_suffix}") is not None
            and getattr(self, f"regridded_precip1{attr_suffix}") is not None
        ):
            # Calculate where we have missing data in either the previous or next forcing dataset.
            ind1Ndv = np.where(
                getattr(self, f"regridded_precip1{attr_suffix}") == self.globalNdv
            )
            ind2Ndv = np.where(
                getattr(self, f"regridded_precip2{attr_suffix}") == self.globalNdv
            )

            arr_set_arr(
                getattr(self, f"final_supp_precip{attr_suffix}"),
                (
                    (
                        getattr(self, f"regridded_precip1{attr_suffix}")[
                            (slice(None),) * ndim
                        ]
                        * weight1
                    )
                    + (
                        getattr(self, f"regridded_precip2{attr_suffix}")[
                            (slice(None),) * ndim
                        ]
                        * weight2
                    )
                ),
                ndim,
            )

            # Set any pixel cells that were missing for either window to missing value.
            getattr(self, f"final_supp_precip{attr_suffix}")[ind1Ndv] = self.globalNdv
            getattr(self, f"final_supp_precip{attr_suffix}")[ind2Ndv] = self.globalNdv

            # Reset for memory efficiency.
            ind1Ndv = None
            ind2Ndv = None
        else:
            # We have missing files.
            arr_set_scalar(
                getattr(self, f"final_supp_precip{attr_suffix}"), self.globalNdv, ndim
            )

    def _gfs_pcp_time_interp(self) -> Any | tuple[Any, Any]:
        """Perform the business logic of the GFS time interpolation.

        Business logic and comments from original code:
            https://github.com/NGWPC/ngen-forcing/blob/a0f217f06a0045d9f139bfa14abe711fc6f248b0/NextGen_Forcings_Engine_BMI/NextGen_Forcings_Engine/core/timeInterpMod.py#L679-L881
        """
        also_elem = {
            "gridded": False,
            "unstructured": True,
            "hydrofabric": False,
        }[self.config_options.grid_type]

        if self.input_forcings.fcst_hour2 <= 120:
            if self.input_forcings.fcst_hour2 % 6 == 1:
                # We are on the first hour of a six-hour period. We can treat
                # the precipitation rate as instantaneous for this hour.
                instPcpGlobal = self.input_forcings.globalPcpRate2
                total1 = None
                total2 = None
                total1_elem = None
                total2_elem = None
            else:
                # We need to calculate the difference from the previous
                # avg window to get an instantaneous value for this hour.
                total1 = self.input_forcings.globalPcpRate1 * (
                    3600.0 * (self.input_forcings.fcst_hour1 % 6)
                )
                if also_elem:
                    total1_elem = self.input_forcings.globalPcpRate1_elem * (
                        3600.0 * (self.input_forcings.fcst_hour1 % 6)
                    )
                if self.input_forcings.fcst_hour2 % 6 == 0:
                    # We have a 0-6, 6-12, 12-18 avg rate....
                    total2 = self.input_forcings.globalPcpRate2 * (3600.0 * 6)
                    if also_elem:
                        total2_elem = self.input_forcings.globalPcpRate2_elem * (
                            3600.0 * 6
                        )
                else:
                    # We have 0-5, 0-4, etc
                    total2 = self.input_forcings.globalPcpRate2 * (
                        3600.0 * (self.input_forcings.fcst_hour2 % 6)
                    )
                    if also_elem:
                        total2_elem = self.input_forcings.globalPcpRate2_elem * (
                            3600.0 * (self.input_forcings.fcst_hour2 % 6)
                        )
                instPcpGlobal = (total2 - total1) / 3600.0
                if also_elem:
                    instPcpGlobal_elem = (total2_elem - total1_elem) / 3600.0
                # Reset variables to free up memory
                total1 = None
                total2 = None
                total1_elem = None
                total2_elem = None
        else:
            # We are in Situation #2 which currently runs out until the end of the
            # end of the GFS forecast cycle of 384 hours.
            if self.input_forcings.fcst_hour2 % 6 == 3:
                # We are on the first 3 hours of a six hour period. Simply treat the average
                # precipitation rate for this time period as the instantaneous precipitation
                # rate.
                instPcpGlobal = self.input_forcings.globalPcpRate2
                if also_elem:
                    instPcpGlobal_elem = self.input_forcings.globalPcpRate2_elem
                total2 = None
                total1 = None
                total1_elem = None
                total2_elem = None
            else:
                total1 = self.input_forcings.globalPcpRate1 * (3600.0 * 3.0)
                total2 = self.input_forcings.globalPcpRate2 * (3600.0 * 6.0)
                if also_elem:
                    total1_elem = self.input_forcings.globalPcpRate1_elem * (
                        3600.0 * 3.0
                    )
                    total2_elem = self.input_forcings.globalPcpRate2_elem * (
                        3600.0 * 6.0
                    )
                instPcpGlobal = (total2 - total1) / (3600.0 * 3.0)
                if also_elem:
                    instPcpGlobal_elem = (total2_elem - total1_elem) / (3600.0 * 3.0)
                # Reset variables to free up memory.
                total1 = None
                total2 = None
                total1_elem = None
                total2_elem = None
        # Return the interpolated grid back to the regridding program.

        # Set any negative values to 0.0
        instPcpGlobal[np.where(instPcpGlobal < 0.0)] = 0.0
        if also_elem:
            instPcpGlobal_elem[np.where(instPcpGlobal_elem < 0.0)] = 0.0

        if not also_elem:
            return instPcpGlobal
        else:
            return instPcpGlobal, instPcpGlobal_elem


def no_interpolation(
    input_forcings: InputForcings, config_options: ConfigOptions, mpi_config: MpiConfig
):
    """No temporal interpolation.

    Function for simply setting the final regridded fields to the
    input forcings that are from the next input forcing frequency.
    :param input_forcings:
    :param config_options:
    :param mpi_config:
    :return:
    """
    _TimeInterp(input_forcings, None, config_options, mpi_config)._no_interpolation()


def no_interpolation_supp_pcp(
    supplemental_precip: SupplementalPrecip,
    config_options: ConfigOptions,
    mpi_config: MpiConfig,
):
    """No temporal interpolation for supplemental precipitation.

    Function for simply setting the final regridded supplemental precipitation
    to the supplemental precipitation grids from the next precip frequency that
    is available.
    :param supplemental_precip:
    :param config_options:
    :param mpi_config:
    :return:
    """
    _TimeInterp(
        None, supplemental_precip, config_options, mpi_config
    )._no_interpolation_supp_pcp()


def nearest_neighbor(
    input_forcings: InputForcings, config_options: ConfigOptions, mpi_config: MpiConfig
):
    """Nearest neighbor temporal interpolation.

    Function for setting the current output regridded forcings to the nearest
    input forecast step.
    :param input_forcings:
    :param config_options:
    :param mpi_config:
    :return:
    """
    _TimeInterp(input_forcings, None, config_options, mpi_config)._nearest_neighbor()


def nearest_neighbor_supp_pcp(
    supplemental_precip: SupplementalPrecip,
    config_options: ConfigOptions,
    mpi_config: MpiConfig,
):
    """Nearest neighbor temporal interpolation for supplemental precipitation.

    Function for setting the current output regridded supplemental precipitation
    to the nearest supplemental precipitation input step.
    :param supplemental_precip:
    :param config_options:
    :param mpi_config:
    :return:
    """
    _TimeInterp(
        None, supplemental_precip, config_options, mpi_config
    )._nearest_neighbor_supp_pcp()


def weighted_average(
    input_forcings: InputForcings, config_options: ConfigOptions, mpi_config: MpiConfig
):
    """Weighted average temporal interpolation for supplemental precipitation.

    Function for setting the current output regridded fields as a weighted
    average between the previous output step and the next output step.
    :param input_forcings:
    :param config_options:
    :param mpi_config:
    :return:
    """
    _TimeInterp(input_forcings, None, config_options, mpi_config)._weighted_average()


def weighted_average_supp_pcp(
    supplemental_precip: SupplementalPrecip,
    config_options: ConfigOptions,
    mpi_config: MpiConfig,
):
    """Weighted average temporal interpolation for supplemental precipitation.

    Function for setting the current output regridded supplemental precipitation fields
    as an average between the previous and next input supplemental precipitation timesteps.
    :param supplemental_precip:
    :param config_options:
    :param mpi_config:
    :return:
    """
    interpolator = _TimeInterp(None, supplemental_precip, config_options, mpi_config)

    weight1, weight2 = interpolator.__calc_weights_for_supp_pcp()

    interpolator._weighted_average_supp_pcp(weight1, weight2)
    if config_options.grid_type == "unstructured":
        interpolator._weighted_average_supp_pcp(weight1, weight2, "_elem")


def gfs_pcp_time_interp(
    input_forcings: InputForcings, config_options: ConfigOptions, mpi_config: MpiConfig
) -> Any | tuple[Any, Any]:
    """Calculate instantaneous precipitation rate from GFS average rates.

    Function that will calculate an instantaneous precipitation rate, representative
    of the latest forecast GFS hour, or range of GFS forecast hours. This is
    done as GFS has a quirky way of outputting precipitation rates.
    :param input_forcings:
    :param config_options:
    :param mpi_config:
    :return: instPcpGlobal
    """
    # There is a chain of logic we will follow here for GFS data.
    # Forecast Hours <= 120:
    # 1.) For first hour in every six hour period, the avg precip rate
    #     will be treated as the instantaneous rate. No processing
    #     needed. So 0-1, 6-7, 12-13, etc.
    # 2.) For remaining hours, we need to calculate the difference between
    #     this avg precip rate, and the previous one. So,
    #     [(0-4)-(0-3)] gives us a precipitation rate for hour 4.
    # Forecast Hours > 120:
    # 1.) For the first three hours of every six hour period, we
    #     will treat the avg precip rate coming in as the instantaneous
    #     value at hour 3. This is because there are no values for
    #     individual hours within this horizon. So, 120-123, 126-129, etc
    # 2.) For other three hours, we need to calculate the difference
    #     between the previous three hourly average rate, and the average
    #     rate for this entire six hour time frame. Why is this the case
    #     with GFS? Ask someone at NCEP.......
    #     [(120-126)-(120-123)] gives us a precipitation rate representative
    #     of the second three hour period. It's not necessarily instantaneous,
    #     but we will treat it as such.
    return _TimeInterp(
        input_forcings, None, config_options, mpi_config
    )._gfs_pcp_time_interp()
