import json
import parcels
from datetime import timedelta
from fieldset_vars import FieldsetVariable
import kernels
import logging
import time
import numpy as np
import xarray as xr
import coastal_boundary

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model.log"),  # Save logs to simulation.log
        logging.StreamHandler()  # Also print logs to console
    ]
)

# Function to load the configuration file
def load_config(config_path="config.json"):
    logging.info("Loading configuration from %s", config_path)
    with open(config_path, "r") as f:
        return json.load(f)

# Function to collect extra variables required by the kernels
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

# Function to dynamically load kernels from the config
def load_kernels(kernel_names, parcels, kernels):
    kernel_list = []
    for kernel_name in kernel_names:
        if hasattr(parcels, kernel_name):
            kernel_list.append(getattr(parcels, kernel_name))
        elif hasattr(kernels, kernel_name):
            kernel_list.append(getattr(kernels, kernel_name))
        else:
            raise ValueError(f"Kernel {kernel_name} not found in parcels or kernels.py.")
    logging.info("Loaded kernels: %s", kernel_names)
    return kernel_list

def create_fieldset(config):
    if config.get("variable_names", {}):
        vars = config["variable_names"]
    
    if config.get("dimension_names", {}):
        dims = config["dimension_names"]
    
    if config.get("coastal_boundary", {}).get("include", False):
        logging.info("Including a coastal boundary.")

        if not config.get("coastal_boundary", {}).get("path", "") == None:
            logging.warning("Coastal boundary has not yet been added to the main dataset. Adding...")
            ds = xr.open_dataset(config["coastal_boundary"]["path"])
            ds_boundary = xr.open_dataset(config["coastal_boundary"]["path"])
            ds_combined = coastal_boundary.add_datasets(ds, ds_boundary)

            fieldset = parcels.FieldSet.from_xarray_dataset(ds_combined, vars, dims)
            logging.info("Fieldset created with filenames: %s", config["coastal_boundary"]["path"])
            
        if config.get("coastal_boundary", {}).get("path", "") == None:
            logging.warning("No coastal boundary path provided. Assuming main path includes the coastal boundary already.")

            fieldset = parcels.FieldSet.from_netcdf(config["pathname"], vars, dims)
            logging.info("Fieldset created with filenames: %s", config["pathname"])
    else:
        logging.info("Not including a coastal boundary.")

        fieldset = parcels.FieldSet.from_netcdf(config["pathname"], vars, dims)
        logging.info("Fieldset created with filenames: %s", config["pathname"])

    return fieldset

def load_coords(config):
    """Load longitude and latitude lists from specified text files in the config."""
    lat_file = config["particles"]["lat_file"]
    lon_file = config["particles"]["lon_file"]

    try:
        # Read the latitude values
        lats = np.loadtxt(lat_file).tolist()  # Convert to list
        logging.info(f"Loaded latitudes: {lats}")

        # Read the longitude values
        lons = np.loadtxt(lon_file).tolist()  # Convert to list
        logging.info(f"Loaded longitudes: {lons}")

        return lats, lons
    except Exception as e:
        logging.error(f"Error loading coordinates: {e}")
        return [], []

def create_particle_set(fieldset, particle_class, config):
    """Create a ParticleSet with initial locations from config."""
    # Load lons and lats from the specified files
    lats, lons = load_coords(config)
    
    if not lons or not lats:
        logging.error("No valid coordinates found. ParticleSet cannot be created.")
        return None  # Return None or raise an exception if coordinates are invalid

    logging.info("Creating ParticleSet with initial locations: lon=%s, lat=%s",
                 lons, lats)

    return parcels.ParticleSet(
        fieldset=fieldset,
        pclass=particle_class,
        lon=lons,
        lat=lats
    )

# Function to set up the ParticleFile for output
def setup_output_file(pset, config):
    output_dir = config["output"]["directory"]
    file_name = config["output"]["file_name"]
    output_interval = timedelta(hours=config["output"].get("output_interval_hours", 6))

    logging.info("Setting up output file: %s/%s with interval %s", output_dir, file_name, output_interval)
    return pset.ParticleFile(name=f"{output_dir}/{file_name}", outputdt=output_interval)

# Main function to execute the simulation
def main():

    # load the model configuration
    config = load_config()

    # Load the kernels dynamically based on the config
    kernel_list = load_kernels(config.get("kernels", []), parcels, kernels)

    # Collect the extra variables required by the kernels
    extra_vars = collect_extra_vars(kernel_list)

    # Add extra particle type variables to particle class
    extra_particle_vars = []
    for extra_var in extra_vars:
        if type(extra_var) == parcels.particle.Variable:
            extra_particle_vars.append(extra_var)
    particles = parcels.JITParticle.add_variables(extra_particle_vars)

    # Create the fieldset based on the provided config
    fieldset = create_fieldset(config)

    # Add extra fieldset type variables to fieldset object
    for extra_var in extra_vars:
        if isinstance(extra_var, FieldsetVariable):
            fieldset.add_constant(extra_var.name, extra_var.value(fieldset))
        else:
            fieldset.add_constant(extra_var.name, extra_var)

    # Create the ParticleSet based on the fieldset object and particle class
    pset = create_particle_set(fieldset, particles, config)

    # Set up the output file
    output_file = setup_output_file(pset, config)

    # Execute the simulation and measure time
    start_time = time.time()
    logging.info(f"Starting the simulation at {start_time}...")
    
    pset.execute(
        kernel_list,
        runtime=timedelta(days=config["runtime"].get("days", 10)),
        dt=timedelta(minutes=config["runtime"].get("dt_minutes", 5)),
        output_file=output_file,
    )

    end_time = time.time()
    logging.info(f"Simulation completed in {end_time - start_time} seconds")

# Check if the script is being run directly
if __name__ == "__main__":
    main()