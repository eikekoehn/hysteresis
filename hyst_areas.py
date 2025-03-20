"""
author: Eike E. Köhn
date: Mar 19, 2025
description: This module contains a number of functions in order to calculate hysteresis area for a given variable.
"""

import numpy as np
import warnings
import xarray as xr
import dask


def calc_hysteresis_area_1D(ref_axis, data_series, nsteps=1000, normalizer='min_max_diff_full_cycle',return_interpolated_vectors = False):
    """
    Calculate the hysteresis area between ramp-up and ramp-down curves.
    
    Parameters:
    ref_axis (array-like): Reference axis values, e.g. atm CO2, GMST, ...
    data_series (array-like): Corresponding data values.
    nsteps (int): Number of interpolation steps.
    normalizer (str): Normalization method ('min_max_diff_rampup', 'min_max_diff_full_cycle', or None).
    
    Note that the ref_axis has to be strictly monotonically increasing during the rampup and strictly monotonically decreasing during the rampdown

    Returns:
    tuple: (hysteresis_area, signed_hysteresis_area, normalized_hysteresis_area, normalizer_value)
    """

    # Handle NaN values in the data
    if np.any(np.isnan(data_series)) or np.any(np.isnan(ref_axis)):
        return np.NaN, np.NaN, np.NaN, np.NaN

    # If nsteps is too small, return nans
    if nsteps < 20: # 20 chosen manually
        warnings.warn("nsteps must be at least 20 for meaningful interpolation.", RuntimeWarning)
        return np.NaN, np.NaN, np.NaN, np.NaN

    # Find min, max, and index of max in the reference axis
    ref_axis_min = np.nanmin(ref_axis)
    ref_axis_max = np.nanmax(ref_axis)
    ref_axis_argmax = np.nanargmax(ref_axis)
    
    # Give a warning if there is no single clear peak in the reference axis (e.g. atm CO2, or GMST)
    if not np.all(np.diff(ref_axis[:ref_axis_argmax+1]) > 0) or not np.all(np.diff(ref_axis[ref_axis_argmax:]) < 0):
        warnings.warn("Reference axis does not have a single clear peak. Results may be unreliable.", RuntimeWarning)

    # Generate a linearly spaced ramping vector
    ramping_vector = np.linspace(ref_axis_min, ref_axis_max, nsteps)
    stepwidth = ramping_vector[1] - ramping_vector[0]

    # Extract and sort ramp-up segment
    rampup_ref_axis = ref_axis[:ref_axis_argmax+1]
    rampup_data_series = data_series[:ref_axis_argmax+1]

    # Extract and sort ramp-down segment (ensure unique values)
    rampdown_ref_axis = ref_axis[ref_axis_argmax:][::-1]
    rampdown_data_series = data_series[ref_axis_argmax:][::-1]

    # Making sure that 2 or more values remain in the reference axis
    if len(rampup_ref_axis) < 2 or len(rampdown_ref_axis) < 2:
        return np.NaN, np.NaN, np.NaN, np.NaN

    # Interpolate data for ramp-up and ramp-down
    interpolated_rampup = np.interp(ramping_vector, rampup_ref_axis, rampup_data_series,left=np.NaN,right=np.NaN)
    interpolated_rampdown = np.interp(ramping_vector, rampdown_ref_axis, rampdown_data_series,left=np.NaN,right=np.NaN)
    
    # Compute absolute difference and area
    difference = np.abs(interpolated_rampup - interpolated_rampdown)
    hysteresis_area = np.nansum(difference * stepwidth)

    # Compute signed hysteresis area (mean of ramp-down minus mean of ramp-up)
    signed_hysteresis_area = np.nanmean(interpolated_rampdown) - np.nanmean(interpolated_rampup)
    
    # Determine normalization factor
    if normalizer == 'min_max_diff_rampup':
        normalizer_value = np.abs(np.nanmax(interpolated_rampup) - np.nanmin(interpolated_rampup))
    elif normalizer == 'min_max_diff_full_cycle':
        max_value = max(np.nanmax(interpolated_rampup), np.nanmax(interpolated_rampdown))
        min_value = min(np.nanmin(interpolated_rampup), np.nanmin(interpolated_rampdown))
        normalizer_value = max_value - min_value
    else:
        normalizer_value = 0
    
    # Compute normalized hysteresis area
    if normalizer_value != 0:
        normalized_hysteresis_area = hysteresis_area / normalizer_value 
    else:
        warnings.warn("Normalization factor is zero; normalized hysteresis area will be NaN.", RuntimeWarning)
        normalized_hysteresis_area = np.NaN
    
    if return_interpolated_vectors == True:
        return hysteresis_area, signed_hysteresis_area, normalized_hysteresis_area, normalizer_value, interpolated_rampup, interpolated_rampdown, ramping_vector
    else:
        return hysteresis_area, signed_hysteresis_area, normalized_hysteresis_area, normalizer_value



def calc_hysteresis_area_3D(ref_axis, da, nsteps=1000, normalizer='min_max_diff_full_cycle', return_interpolated_vectors=False):
    """
    Compute hysteresis at multiple locations in a 3D xarray DataArray.

    Parameters:
    ref_axis (array-like): Reference axis values.
    da (xarray.DataArray): 3D data array with dimensions (year, lat, lon).
    nsteps (int): Number of interpolation steps.
    normalizer (str): Normalization method.
    return_interpolated_vectors (bool): If True, returns interpolated vectors as well.

    Returns:
    xarray.Dataset: Contains hysteresis area, signed hysteresis area, 
                    normalized hysteresis area, and normalizer values.
    """

    def hysteresis_wrapper(data_series):
        """Wrapper to apply calc_hysteresis_area_1D on 1D data."""
        if np.any(np.isnan(data_series)):  # Skip if any NaN
            return np.NaN, np.NaN, np.NaN, np.NaN  # Ensure correct shape
        return calc_hysteresis_area_1D(ref_axis, data_series, nsteps, normalizer, return_interpolated_vectors)[:4]

    # Apply function across lat/lon using vectorized approach
    results = xr.apply_ufunc(
        hysteresis_wrapper, da,
        input_core_dims=[["year"]],  # Apply along 'year' axis
        output_core_dims=[[], [], [], []],  # Each output is a scalar per lat/lon
        vectorize=True,  # Automatically loops over lat/lon
        dask="parallelized",  # Enables parallelization
        output_dtypes=[float, float, float, float]  # Ensure correct output types
    )

    # Ensure results retain (lat, lon) structure
    dataset = xr.Dataset(
        {
            "hysteresis_area": (["lat", "lon"], results[0].data),
            "signed_hysteresis_area": (["lat", "lon"], results[1].data),
            "normalized_hysteresis_area": (["lat", "lon"], results[2].data),
            "normalizer_value": (["lat", "lon"], results[3].data)
        },
        coords={"lat": da.lat, "lon": da.lon}
    )

    return dataset
