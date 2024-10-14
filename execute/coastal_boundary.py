import xarray as xr
import numpy  as np

def coastal_boundary(ds, 
                     var_names={'U': 'U', 'V': 'V'}, 
                     dim_names={'lon': 'lon', 'lat': 'lat', 'time': 'time'},
                     rolling_window=3, threshold=2000):
    """
    Create an artificial coastal boundary from U and V water velocity components at a single time step
    following an example provided by Iury Simoes-Sousa (@iuryt).

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing U and V water velocity components.
    var_names : dict, optional
        Dictionary with keys 'U' and 'V' specifying the variable names for U and V velocity components. 
        Default is {'U': 'U', 'V': 'V'}.
    dim_names : dict, optional
        Dictionary with keys 'lon', 'lat', and 'time' specifying the dimension names in the dataset.
        Default is {'lon': 'lon', 'lat': 'lat', 'time': 'time'}.
    rolling_window : int, optional
        The window size for rolling mean used to smooth the boundary. Default is 3.
    threshold : float, optional
        The threshold for velocity magnitude to define significant boundaries. Default is 2000.

    Returns:
    --------
    boundary : xarray.Dataset
        Dataset containing the smoothed U and V velocity components for the artificial boundary at the single time step.
    norm : xarray.DataArray
        The normalized velocity magnitude used to create the boundary at the single time step.
    """
    ds_timestep = ds.isel({dim_names['time']: 1})

    landmask = (ds_timestep[var_names['U']] - ds_timestep[var_names['U']]).fillna(1)

    u_boundary = -(1e3 * landmask.differentiate(dim_names['lon']))
    v_boundary = -(1e3 * landmask.differentiate(dim_names['lat']))

    norm = np.sqrt(u_boundary**2 + v_boundary**2)
    norm = norm.where(norm > threshold)

    u_boundary = (u_boundary / norm).fillna(0)
    v_boundary = (v_boundary / norm).fillna(0)

    boundary = xr.Dataset({
        var_names['U']: u_boundary,
        var_names['V']: v_boundary
    }).rolling({dim_names['lon']: rolling_window, dim_names['lat']: rolling_window}, center=True).mean()

    return boundary, norm

def dimmentionalize(ds, ds_boundary):
    """
    Upscales a boundary dataset (single time step) to have the same time steps as it's parent dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The original dataset with the desired time steps.
    ds_boundary : xarray.Dataset
        The dataset with artificial boundary data to be upscaled.

    Returns:
    --------
    ds_boundary_upscaled : xarray.Dataset
        The upscaled dataset with the same time steps as ds.
    """
    
    n_time_steps = ds.sizes['time']

    ds_boundary_upscaled = xr.concat([ds_boundary] * n_time_steps, dim='time')

    ds_boundary_upscaled['time'] = ds['time']

    return ds_boundary_upscaled