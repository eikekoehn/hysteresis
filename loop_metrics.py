"""
author: Eike E. Köhn
date: Mar 19, 2025
description: This module contains a number of functions in order to calculate the types of loops
"""

import numpy as np
import warnings
import xarray as xr
import dask
import matplotlib.pyplot as plt


def calc_loop_gap_1D(ref_axis, data_series, nsteps=1000, n_ave=1, normalizer='min_max_diff_full_cycle', return_interpolated_vectors=False):
    """
    Calculate the difference between the start and end of the data series after averaging the values
    at the beginning and end. Normalize this difference by the min-max difference of the entire series.
    
    Parameters:
    ref_axis (array-like): Reference axis values (e.g., time, temperature, etc.).
    data_series (array-like): Corresponding data values (e.g., atmospheric CO2, GMST, etc.).
    nsteps (int, optional): Number of interpolation steps for the ramp-up and ramp-down phases. Default is 1000.
    n_ave (int, optional): Number of data points to average over at the start and end of the series. Default is 1.
    return_interpolated_vectors (bool, optional): If True, return the interpolated ramp-up and ramp-down vectors. Default is False.
    
    Returns:
    tuple: A tuple containing the following:
        - `start_end_difference` (float): The difference between the averaged start and end values of the data series.
        - `normalized_start_end_difference` (float): The normalized difference, divided by the min-max range of the entire series.
        - `normalizer` (float): The min-max difference of the entire series (used for normalization).
        - `interpolated_rampup` (optional, array-like): The interpolated ramp-up data series.
        - `interpolated_rampdown` (optional, array-like): The interpolated ramp-down data series.
        - `ramping_vector` (optional, array-like): The linearly spaced reference axis for interpolation.
    """

    # Check for NaN values in the data
    if np.any(np.isnan(data_series)) or np.any(np.isnan(ref_axis)):
        return np.NaN, np.NaN, np.NaN

    # Validate that n_ave is not too large (should not exceed half of the reference axis size)
    if n_ave > np.floor(np.size(ref_axis) / 2):
        warnings.warn("n_ave must be smaller than half the size of the reference axis for meaningful results.", RuntimeWarning)
        return np.NaN, np.NaN, np.NaN

    # Find min, max, and index of maximum in the reference axis (assuming ramp-up then ramp-down behavior)
    ref_axis_min = np.nanmin(ref_axis)
    ref_axis_max = np.nanmax(ref_axis)
    ref_axis_argmax = np.nanargmax(ref_axis)
    
    # Warning if the reference axis doesn't have a clear ramp-up and ramp-down (no single peak)
    if not np.all(np.diff(ref_axis[:ref_axis_argmax+1]) > 0) or not np.all(np.diff(ref_axis[ref_axis_argmax:]) < 0):
        warnings.warn("Reference axis does not have a single clear peak. Results may be unreliable.", RuntimeWarning)

    # Generate a linearly spaced ramping vector for interpolation
    ramping_vector = np.linspace(ref_axis_min, ref_axis_max, nsteps)

    # Extract the ramp-up and ramp-down segments
    rampup_ref_axis = ref_axis[:ref_axis_argmax+1]
    rampup_data_series = data_series[:ref_axis_argmax+1]
    
    rampdown_ref_axis = ref_axis[ref_axis_argmax:][::-1]  # Reverse ramp-down phase
    rampdown_data_series = data_series[ref_axis_argmax:][::-1]

    # Ensure there are at least two values for both ramp-up and ramp-down
    if len(rampup_ref_axis) < 2 or len(rampdown_ref_axis) < 2:
        return np.NaN, np.NaN, np.NaN

    # Interpolate data for ramp-up and ramp-down phases
    interpolated_rampup = np.interp(ramping_vector, rampup_ref_axis, rampup_data_series, left=np.NaN, right=np.NaN)
    interpolated_rampdown = np.interp(ramping_vector, rampdown_ref_axis, rampdown_data_series, left=np.NaN, right=np.NaN)

    # Mask out NaN values in both interpolated ramp-up and ramp-down vectors
    mask = ~np.logical_or(np.isnan(interpolated_rampup), np.isnan(interpolated_rampdown))
    mask2 = mask * 1.0  # Convert Boolean mask to float (1.0 for valid data, NaN for invalid data)
    mask2[mask2 == 0.0] = np.NaN

    # Find the first valid index where both ramp-up and ramp-down have data
    minidx = np.nanargmin(mask2)

    # Calculate the average start value (from the ramp-up) and end value (from the ramp-down)
    data_start_value = np.nanmean(interpolated_rampup[minidx:minidx+n_ave])  # Average of the first 'n_ave' points
    data_end_value = np.nanmean(interpolated_rampdown[minidx:minidx+n_ave])  # Average of the last 'n_ave' points

    # Calculate the difference between the start and end values
    start_end_difference = data_end_value - data_start_value

    # Determine normalization factor
    if normalizer == 'min_max_diff_rampup':
        normalizer_value = np.abs(np.nanmax(interpolated_rampup) - np.nanmin(interpolated_rampup))
    elif normalizer == 'min_max_diff_full_cycle':
        max_value = max(np.nanmax(interpolated_rampup), np.nanmax(interpolated_rampdown))
        min_value = min(np.nanmin(interpolated_rampup), np.nanmin(interpolated_rampdown))
        normalizer_value = max_value - min_value
    else:
        normalizer_value = 0

    # Normalize the difference, handling the case where the normalizer is 0
    if normalizer_value != 0:
        normalized_start_end_difference = start_end_difference / normalizer_value
    else:
        normalized_start_end_difference = np.NaN  # If normalizer_value is 0 (constant data), return NaN

    # Optionally return the interpolated ramp-up and ramp-down vectors and the ramping vector
    if return_interpolated_vectors:
        return start_end_difference, normalized_start_end_difference, normalizer_value, interpolated_rampup, interpolated_rampdown, ramping_vector
    else:
        return start_end_difference, normalized_start_end_difference, normalizer_value




def calc_loop_gap_3D(ref_axis, da, nsteps=1000, n_ave=1, normalizer='min_max_diff_full_cycle', return_interpolated_vectors=False):
    """
    Compute hysteresis metrics (start-end difference, normalized difference) across multiple lat/lon locations 
    in a 3D `xarray.DataArray` using a reference axis (e.g., time, temperature).

    This function applies the `calc_loop_gap_1D` function to each lat/lon point, calculating the difference 
    between the start and end of the data series after averaging over the first `n_ave` and last `n_ave` data points. 
    The difference is normalized using the min-max range of the series.

    Parameters:
    ref_axis (array-like): 1D array representing the reference axis values (e.g., time, temperature, etc.).
    da (xarray.DataArray): 3D data array with dimensions (year, lat, lon). The `year` dimension represents the data 
                            values, while the `lat` and `lon` dimensions correspond to the geographical grid points.
    nsteps (int, optional): Number of interpolation steps used to align the ramp-up and ramp-down phases of the reference axis.
                             Default is 1000.
    n_ave (int, optional): Number of data points at the start and end to average over to calculate the start and end values 
                            of the data series. Default is 1.
    return_interpolated_vectors (bool, optional): If True, returns the interpolated ramp-up and ramp-down vectors. 
                                                  Default is False.

    Returns:
    xarray.Dataset: A dataset containing:
        - `start_end_difference` (float): The difference between the averaged start and end values of the data series 
          for each lat/lon point.
        - `normalized_start_end_difference` (float): The normalized difference, divided by the min-max range of the 
          entire data series, for each lat/lon point.
        - `normalizer_value` (float): The min-max difference of the entire data series, used for normalization, for each lat/lon point.

    If `return_interpolated_vectors=True`, the function also returns:
        - `interpolated_rampup` (array): Interpolated ramp-up values.
        - `interpolated_rampdown` (array): Interpolated ramp-down values.
        - `ramping_vector` (array): The reference axis values used for interpolation.
    
    Notes:
    - The `ref_axis` should be strictly monotonically increasing during the ramp-up phase and strictly monotonically decreasing 
      during the ramp-down phase.
    - The function is designed to work efficiently with large datasets by using `xarray`'s parallel processing capabilities 
      (via `dask`).
    """

    def loop_gap_wrapper(data_series):
        """Wrapper to apply calc_loop_gap_1D on 1D data."""
        if np.any(np.isnan(data_series)):  # Skip if any NaN
            return np.NaN, np.NaN, np.NaN  # Ensure correct shape
        return calc_loop_gap_1D(ref_axis, data_series, nsteps, n_ave, normalizer, return_interpolated_vectors)[:3]

    # Apply function across lat/lon using vectorized approach
    results = xr.apply_ufunc(
        loop_gap_wrapper, da,
        input_core_dims=[["year"]],  # Apply along 'year' axis
        output_core_dims=[[], [], []],  # Each output is a scalar per lat/lon
        vectorize=True,  # Automatically loops over lat/lon
        dask="parallelized",  # Enables parallelization
        output_dtypes=[float, float, float]  # Ensure correct output types
    )

    # Ensure results retain (lat, lon) structure
    dataset = xr.Dataset(
        {
            "start_end_difference": (["lat", "lon"], results[0].data),
            "normalized_start_end_difference": (["lat", "lon"], results[1].data),
            "normalizer_value": (["lat", "lon"], results[2].data),
        },
        coords={"lat": da.lat, "lon": da.lon}
    )

    return dataset

