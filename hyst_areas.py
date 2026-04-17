"""
author: Eike E. Köhn
date: Mar 19, 2025
description: This module contains a number of functions in order to calculate hysteresis area for a given variable.
"""

import numpy as np
import warnings
import xarray as xr
import dask
from numba import njit


def calc_hysteresis_area_1D(ref_axis, data_series, nsteps=1000, normalizer='min_max_diff_full_cycle', return_interpolated_vectors = False, normalize_by_ref_axis_range = False):
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

    # Normalize normalized hysteresis area with ref_axis range?
    if normalize_by_ref_axis_range == True:
        ref_axis_min = min( np.min(rampup_ref_axis), np.min(rampdown_ref_axis))
        ref_axis_max = max( np.max(rampup_ref_axis), np.max(rampdown_ref_axis))
        ref_axis_range = np.abs( ref_axis_max - ref_axis_min )
        normalized_hysteresis_area = normalized_hysteresis_area/ref_axis_range
        hysteresis_area = hysteresis_area/ref_axis_range
        #print(f'The ref_axis range is: {ref_axis_range}')

    if return_interpolated_vectors == True:
        return hysteresis_area, signed_hysteresis_area, normalized_hysteresis_area, normalizer_value, interpolated_rampup, interpolated_rampdown, ramping_vector 
    else:
        return hysteresis_area, signed_hysteresis_area, normalized_hysteresis_area, normalizer_value



def calc_hysteresis_area_3D(ref_axis, da, nsteps=1000, normalizer='min_max_diff_full_cycle', return_interpolated_vectors=False, normalize_by_ref_axis_range = False):
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

        # Proceed with calculation
        result = calc_hysteresis_area_1D(
            ref_axis,
            data_series,
            nsteps=nsteps,
            normalizer=normalizer,
            return_interpolated_vectors=return_interpolated_vectors,
            normalize_by_ref_axis_range=normalize_by_ref_axis_range
        )

        # Fallback in case of unexpected result structure
        if return_interpolated_vectors:
           if len(result) != 7:
               nan_vector = np.full(nsteps, np.nan)
               return np.NaN, np.NaN, np.NaN, np.NaN, nan_vector, nan_vector, nan_vector
        else:
           if len(result) != 4:
               return np.NaN, np.NaN, np.NaN, np.NaN
        
        return result

    if return_interpolated_vectors==False:
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

    elif return_interpolated_vectors==True:
        # Apply function across lat/lon using vectorized approach
        results = xr.apply_ufunc(
            hysteresis_wrapper, da,
            input_core_dims=[["year"]],  # Apply along 'year' axis
            output_core_dims=[[], [], [], [], ["vec_unit"], ["vec_unit"], ["vec_unit"]],  # Each output is a scalar per lat/lon
            vectorize=True,  # Automatically loops over lat/lon
            dask="parallelized",  # Enables parallelization
            output_dtypes=[float, float, float, float, float, float, float]  # Ensure correct output types
        )
        
        # Ensure results retain (lat, lon) structure
        dataset = xr.Dataset(
            {
                "hysteresis_area": (["lat", "lon"], results[0].data),
                "signed_hysteresis_area": (["lat", "lon"], results[1].data),
                "normalized_hysteresis_area": (["lat", "lon"], results[2].data),
                "normalizer_value": (["lat", "lon"], results[3].data),
                "interpolated_rampup": (["lat", "lon","vec_unit"], results[4].data),
                "interpolated_rampdown": (["lat", "lon","vec_unit"], results[5].data),
                "ramping_vector": (["lat", "lon", "vec_unit"], results[6].data)

            },
            coords={"lat": da.lat, "lon": da.lon, "vec_unit": np.arange(np.shape(results[6].data)[-1])}
        )

    return dataset












@njit
def interp_numba(x, xp, fp):
    """
    Fast 1D linear interpolation (like np.interp) using Numba.
    Assumes xp is increasing.
    """
    result = np.empty(x.shape, dtype=np.float64)
    for i in range(x.shape[0]):
        xi = x[i]
        if xi < xp[0] or xi > xp[-1]:
            result[i] = np.nan
        else:
            for j in range(xp.shape[0] - 1):
                if xp[j] <= xi <= xp[j + 1]:
                    x0, x1 = xp[j], xp[j + 1]
                    y0, y1 = fp[j], fp[j + 1]
                    result[i] = y0 + (xi - x0) * (y1 - y0) / (x1 - x0)
                    break
    return result


@njit
def safe_interp(x, xp, fp):
    result = np.interp(x, xp, fp)
    for i in range(len(x)):
        if x[i] < xp[0] or x[i] > xp[-1]:
            result[i] = np.nan  # or any default value like -9999
    return result

# Your Numba-compatible 1D interpolation and hysteresis calculation
@njit
def calc_hysteresis_area_1D_numba_norm(ref_axis, data_series, nsteps, normalizer_flag):
    if np.any(np.isnan(data_series)) or np.any(np.isnan(ref_axis)):
        return np.NaN, np.NaN, np.NaN, np.NaN

    if nsteps < 20:
        return np.NaN, np.NaN, np.NaN, np.NaN

    ref_axis_min = np.min(ref_axis)
    ref_axis_max = np.max(ref_axis)
    ref_axis_argmax = np.argmax(ref_axis)

    if not np.all(np.diff(ref_axis[:ref_axis_argmax + 1]) > 0) or not np.all(np.diff(ref_axis[ref_axis_argmax:]) < 0):
        # No warnings in njit mode, so just return NaNs if the axis isn't valid
        return np.NaN, np.NaN, np.NaN, np.NaN

    ramping_vector = np.linspace(ref_axis_min, ref_axis_max, nsteps)
    stepwidth = ramping_vector[1] - ramping_vector[0]

    rampup_ref = ref_axis[:ref_axis_argmax + 1]
    rampup_data = data_series[:ref_axis_argmax + 1]

    rampdown_ref = ref_axis[ref_axis_argmax:][::-1]
    rampdown_data = data_series[ref_axis_argmax:][::-1]

    if len(rampup_ref) < 2 or len(rampdown_ref) < 2:
        return np.NaN, np.NaN, np.NaN, np.NaN

    interp_up = safe_interp(ramping_vector, rampup_ref, rampup_data)#, left=np.NaN, right=np.NaN)
    interp_down = safe_interp(ramping_vector, rampdown_ref, rampdown_data)#, left=np.NaN, right=np.NaN)

    diff = np.abs(interp_up - interp_down)
    hyst_area = np.nansum(diff * stepwidth)
    signed_area = np.nanmean(interp_down) - np.nanmean(interp_up)

    if normalizer_flag == 1:  # rampup min-max
        normalizer_value = np.abs(np.nanmax(interp_up) - np.nanmin(interp_up))
    elif normalizer_flag == 2:  # full cycle min-max
        normalizer_value = np.nanmax([np.nanmax(interp_up), np.nanmax(interp_down)]) - np.nanmin([np.nanmin(interp_up), np.nanmin(interp_down)])
    else:
        normalizer_value = 0.0

    norm_area = hyst_area / normalizer_value if normalizer_value != 0.0 else np.NaN

    return hyst_area, signed_area, norm_area, normalizer_value



def calc_hysteresis_area_3D_numba(ref_axis, da, nsteps=1000, normalizer='min_max_diff_full_cycle'):
    """
    Apply 1D hysteresis area calculation to 3D xarray DataArray using Numba.
    """
    # Convert normalizer string to integer flag
    normalizer_map = {
        None: 0,
        'min_max_diff_rampup': 1,
        'min_max_diff_full_cycle': 2
    }
    normalizer_flag = normalizer_map.get(normalizer, 2)

    # Ensure ref_axis is a NumPy array
    ref_axis_np = np.asarray(ref_axis)

    def wrapped_numpy(data_series):
        # Important: make sure data passed to Numba is a NumPy array
        return calc_hysteresis_area_1D_numba_norm(
            ref_axis_np,
            np.asarray(data_series),
            nsteps,
            normalizer_flag
        )

    # Apply across all non-time dims
    result = xr.apply_ufunc(
        wrapped_numpy,
        da,
        input_core_dims=[["year"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, float]
    )

    return xr.Dataset(
        {
            "hysteresis_area": result[0],
            "signed_hysteresis_area": result[1],
            "normalized_hysteresis_area": result[2],
            "normalizer_value": result[3],
        },
        coords={k: v for k, v in da.coords.items() if k != "year"}
    )
