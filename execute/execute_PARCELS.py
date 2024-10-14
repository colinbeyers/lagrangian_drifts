import sys
import os
import logging

import json

from   datetime import timedelta
import time

import numpy as np
import xarray as xr

import parcels
import coastal_boundary
import kernels
from   fieldset_vars import FieldsetVariable

# Temporary log-to-consol configuration
logging.basicConfig(
    level       = logging.INFO,
    format      = '%(asctime)s - %(levelname)s - %(message)s',
    handlers    = [logging.StreamHandler()]
)

def load_config(config_path=None):
    if config_path is None:
        logging.error("No configuration file path provided.")
        sys.exit(1)
    
    logging.info(f"Loading configuration from {config_path}")
    
    if not os.path.exists(config_path):
        logging.error(f"Configuration file {config_path} does not exist.")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def collect_extra_vars(kernel_list):
    extra_vars = []
    for kernel in kernel_list:
        var_name = f"{kernel.__name__}_vars"
        if hasattr(kernels, var_name):
            kernel_vars = getattr(kernels, var_name)
            if isinstance(kernel_vars, dict):
                extra_vars.extend(kernel_vars.values())  # Collect FieldsetVariable objects
            else:
                extra_vars.extend(kernel_vars)
    logging.info("Collected extra variables for kernels: %s", extra_vars)
    return extra_vars

def load_kernels(kernel_names, parcels, kernels):
    """
    Checks to see if the called kernels exist and then adds it to the list.
    """
    kernel_list = []

    # check if each kernel exists in PARCELS or custom file
    for kernel_name in kernel_names:
        if hasattr(parcels, kernel_name):
            kernel_list.append(getattr(parcels, kernel_name))
        elif hasattr(kernels, kernel_name):
            kernel_list.append(getattr(kernels, kernel_name))
        else:
            logging.error(f"Kernel {kernel_name} not found in PARCELS or kernels.py.")
            sys.exit(1)

    logging.info(f"Loaded kernels: {kernel_names}")

    return kernel_list

def create_fieldset(config, extra_vars):
    vars = config["data"]["var_names"]
    dims = config["data"]["dim_names"]
    
    if config["setup"]["coastal_boundary"]["include"] == True:
        logging.info("Including a coastal boundary.")

        if config["setup"]["coastal_boundary"]["path"] != None:
            logging.warning("Coastal boundary has not been added to the main dataset. Adding...")
            ds          = xr.open_dataset(config["data"]["path"])
            boundary    = xr.open_dataset(config["setup"]["coastal_boundary"]["path"])
            ds_boundary = coastal_boundary.dimmentionalize(ds, boundary)
            ds_combined = ds + ds_boundary
            ds_combined = ds_combined.sel(time=slice(config["setup"]["runtime"]["start"], None))

            fieldset = parcels.FieldSet.from_xarray_dataset(ds_combined, vars, dims)
            logging.info(f"Fieldset created with filenames: {config["data"]["path"]}, "
                         + f"{config["setup"]["coastal_boundary"]["path"]}")
            
        if config["setup"]["coastal_boundary"]["path"] == None:
            logging.warning("No coastal boundary path provided."
                            + "Assuming main dataset includes the coastal boundary already.")
            
            ds          = xr.open_dataset(config["data"]["path"])
            ds          = ds.sel(time=slice(config["setup"]["runtime"]["start"], None))
            fieldset    = parcels.FieldSet.from_xarray_dataset(ds, vars, dims)

            logging.info(f"Fieldset created with filenames: {config["data"]["path"]}")
    else:
        logging.warning("Not including a coastal boundary.")

        ds          = xr.open_dataset(config["data"]["path"])
        ds          = ds.sel(time=slice(config["setup"]["runtime"]["start"], None))
        fieldset    = parcels.FieldSet.from_xarray_dataset(ds, vars, dims)

        logging.info(f"Fieldset created with filenames: {config["data"]["path"]}")

    # Add extra fieldset type variables to fieldset object
    for extra_var in extra_vars:
        if isinstance(extra_var, FieldsetVariable):
            fieldset.add_constant(extra_var.name, extra_var.value(fieldset))
        else:
            fieldset.add_constant(extra_var.name, extra_var)

    return fieldset

def load_particle_locs(config):
    """Load longitude and latitude lists from specified text files in the config."""
    lat_file = config["setup"]["particles"]["lat_file"]
    lon_file = config["setup"]["particles"]["lon_file"]

    try:
        # Read the latitude values
        lats = np.loadtxt(lat_file).tolist()
        logging.info(f"Loaded latitudes from {lat_file}.")

        # Read the longitude values
        lons = np.loadtxt(lon_file).tolist()
        logging.info(f"Loaded longitudes from {lon_file}")

        return lats, lons
    
    except Exception as exc:
        logging.error(f"Error loading coordinates: {exc}")
        sys.exit(1)

def setup_particles_class(extra_vars):
    extra_particle_vars = []
    for extra_var in extra_vars:
        if type(extra_var) == parcels.particle.Variable:
            extra_particle_vars.append(extra_var)
    particles = parcels.JITParticle.add_variables(extra_particle_vars)

    logging.info(f"Partical class initialized successfully with extra variables {extra_particle_vars}.")
    return particles

def create_particle_set(fieldset, particle_class, config):
    """Create a ParticleSet with initial locations from config."""
    # Load lons and lats from the specified files
    lats, lons = load_particle_locs(config)
    
    if not lons or not lats:
        logging.error("No valid coordinates found. ParticleSet cannot be created.")
        return None  # Return None or raise an exception if coordinates are invalid

    logging.info("Creating ParticleSet...",
                 lons, lats)

    return parcels.ParticleSet(
        fieldset=fieldset,
        pclass=particle_class,
        lon=lons,
        lat=lats
    )

def setup_output_file(pset, config):
    output_dir      = config["model"]["path"] + 'output/'
    file_name       = config["model"]["name"] + ".zarr"
    output_interval = timedelta(hours=config["output"]["interval_in_hours"])

    logging.info(f"Setting up output file {output_dir}{file_name} with interval {output_interval}.")
    return pset.ParticleFile(name=f"{output_dir}/{file_name}", outputdt=output_interval)

def main():

    # pass the second command line arg to load_config()
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else None)

    # initialize the main log after loading the config file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_path = config["model"]["path"] + config["model"]["name"] + ".log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Configuration {sys.argv[1]} loaded successfully.")

    kernels_list    = load_kernels(config["setup"]["kernels"], parcels, kernels)
    extra_vars      = collect_extra_vars(kernels_list)
    particles       = setup_particles_class(extra_vars)
    fieldset        = create_fieldset(config, extra_vars)
    pset            = create_particle_set(fieldset, particles, config)
    output_file     = setup_output_file(pset, config)

    # Execute the simulation and measure time
    start_time = time.time()
    logging.info(f"Starting the simulation at {start_time}...")
    
    pset.execute(
        kernels_list,
        runtime     = timedelta(days=config["setup"]["runtime"]["days"]),
        dt          = timedelta(minutes=config["setup"]["runtime"]["dt_in_minutes"]),
        output_file = output_file,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Simulation completed in {int(minutes)} minutes, {seconds:.1f} seconds")


# Check if the script is being run directly
if __name__ == "__main__":
    main()