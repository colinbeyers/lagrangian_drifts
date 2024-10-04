import json
import parcels
from datetime import timedelta
from fieldset_vars import FieldsetVariable
import kernels
import logging
import time
import numpy as np
import xarray as xr

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

# Function to create the FieldSet from the dataset and config
def create_fieldset(config):
    # Load the main dataset
    main_dataset = xr.open_dataset(config["pathname"])
    
    # Check if coastal boundary currents should be included
    if config.get("coastal_boundary", {}).get("include", False):
        coastal_boundary_path = config["coastal_boundary"]["path"]
        logging.info("Loading coastal boundary currents from %s", coastal_boundary_path)

        # Load the coastal boundary currents dataset
        coastal_dataset = xr.open_dataset(coastal_boundary_path)

        # Combine the datasets, assuming they have the same dimensions
        combined_dataset = coastal_dataset + main_dataset

        # Assign the combined data variables to the filenames and variables
        filenames = {
            "U": combined_dataset["U"],
            "V": combined_dataset["V"],
        }
        variables = {
            "U": "U",
            "V": "V",
        }
        dimensions = config["dimensions"]

        # Create a FieldSet with the combined dataset
        fieldset = parcels.FieldSet.from_xarray_dataset(combined_dataset, variables, dimensions)
    else:
        logging.info("Creating FieldSet without coastal boundary currents")
        # Create the FieldSet without combining datasets
        fieldset = parcels.FieldSet.from_netcdf(config["pathname"], variables, dimensions)

    logging.info("FieldSet created with filenames: %s", config["pathname"])
    return fieldset

# Function to create the ParticleSet
def create_particle_set(fieldset, particle_class, config):
    logging.info("Creating ParticleSet with initial locations: lon=%s, lat=%s",
                 config["particles"]["lon"], config["particles"]["lat"])
    return parcels.ParticleSet(
        fieldset=fieldset,
        pclass=particle_class,
        lon=config["particles"]["lon"],
        lat=config["particles"]["lat"]
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
    # Load the configuration
    config = load_config()

    # Load the kernels dynamically based on the config
    kernel_list = load_kernels(config.get("kernels", []), parcels, kernels)

    # logging.info(f"Debug: {kernel_list}")

    # Collect the extra variables required by the kernels
    extra_vars = collect_extra_vars(kernel_list)

    extra_particle_vars = []
    for extra_var in extra_vars:
        if type(extra_var) == parcels.particle.Variable:
            extra_particle_vars.append(extra_var)
    particles = parcels.JITParticle.add_variables(extra_particle_vars)

    # Create the fieldset based on the provided config
    fieldset = create_fieldset(config)

    for extra_var in extra_vars:
        if isinstance(extra_var, FieldsetVariable):
            fieldset.add_constant(extra_var.name, extra_var.value(fieldset))  # Pass fieldset here
        else:
            fieldset.add_constant(extra_var.name, extra_var)

    # Create the ParticleSet based on the particles defined in the config
    pset = create_particle_set(fieldset, particles, config)

    # Set up the output file
    output_file = setup_output_file(pset, config)

    # Execute the simulation and measure time
    start_time = time.time()
    logging.info("Starting the simulation...")
    
    pset.execute(
        kernel_list,  # list of kernels to be executed
        runtime=timedelta(days=config["runtime"].get("days", 10)),
        dt=timedelta(minutes=config["runtime"].get("dt_minutes", 5)),
        output_file=output_file,
    )

    end_time = time.time()
    logging.info("Simulation completed in %.2f seconds", end_time - start_time)

# Check if the script is being run directly
if __name__ == "__main__":
    main()