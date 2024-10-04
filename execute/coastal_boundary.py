def coastal_boundary(ds, u_var='U', v_var='V', lon_dim='lon', lat_dim='lat', rolling_window=3, threshold=2000, time_dim='time'):
    """
    Create an artificial coastal boundary from U and V water velocity components at a single time step 
    following an example provided by Iury Simoes-Sousa.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing U and V water velocity components.
    u_var : str, optional
        Name of the U velocity component in the dataset. Default is 'U'.
    v_var : str, optional
        Name of the V velocity component in the dataset. Default is 'V'.
    lon_dim : str, optional
        Name of the longitude dimension in the dataset. Default is 'lon'.
    lat_dim : str, optional
        Name of the latitude dimension in the dataset. Default is 'lat'.
    rolling_window : int, optional
        The window size for rolling mean used to smooth the boundary. Default is 3.
    threshold : float, optional
        The threshold for velocity magnitude to define significant boundaries. Default is 2000.
    time_dim : str, optional
        The name of the time dimension. Default is 'time'.
        
    Returns:
    --------
    boundary : xarray.Dataset
        Dataset containing the smoothed U and V velocity components for the artificial boundary at the single time step.
    norm : xarray.DataArray
        The normalized velocity magnitude used to create the boundary at the single time step.
    """
    # Select the first time step to optimize computation speed
    ds_timestep = ds.isel({time_dim: 1})

    # Create landmask by setting NaN values over land (assuming that land regions have NaN velocities)
    landmask = (ds_timestep[u_var] - ds_timestep[u_var]).fillna(1)

    # Differentiate along longitude and latitude to create artificial boundary velocity components
    u_boundary = -(1e3 * landmask.differentiate(lon_dim))
    v_boundary = -(1e3 * landmask.differentiate(lat_dim))

    # Calculate the magnitude (norm) of the velocity vectors
    norm = np.sqrt(u_boundary**2 + v_boundary**2)

    # Apply the threshold to filter out small velocity magnitudes
    norm = norm.where(norm > threshold)

    # Normalize U and V components
    u_boundary = (u_boundary / norm).fillna(0)
    v_boundary = (v_boundary / norm).fillna(0)

    # Apply a rolling mean to smooth the boundary
    boundary = xr.Dataset({
        u_var: u_boundary,
        v_var: v_boundary
    }).rolling({lon_dim: rolling_window, lat_dim: rolling_window}, center=True).mean()

    return boundary, norm

def dimmentionalize(ds, ds_boundary):
    """
    Upscale ds_boundary to have the same time steps as ds.

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
    
    # Get the number of time steps in the original dataset
    num_time_steps = ds.sizes['time']

    # Create a new dataset with the same time steps as ds
    ds_boundary_upscaled = xr.concat([ds_boundary] * num_time_steps, dim='time')

    # Reset the time dimension to match the original dataset's time coordinates
    ds_boundary_upscaled['time'] = ds['time']

    return ds_boundary_upscaled

def add_datasets(ds1, ds2):
    # Check if the dimensions are the same
    if ds1.dims != ds2.dims:
        raise ValueError("Datasets have different dimensions and cannot be added.")

    # Check if the coordinates match
    if not all(ds1.coords.equals(ds2.coords)):
        raise ValueError("Datasets have different coordinates and cannot be added.")

    # Check if the variables in both datasets are compatible (e.g., numeric types)
    if not all(ds1[var].dtype == ds2[var].dtype for var in ds1.data_vars if var in ds2.data_vars):
        raise TypeError("Datasets contain variables with incompatible data types.")

    # Perform the addition
    return ds1 + ds2