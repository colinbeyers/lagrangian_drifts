import math
import parcels
import numpy as np
from operator import attrgetter

BoundaryCheck_vars = {
    'lat_min': '0',
    'lat_max': '0',
    'lon_min': '0',
    'lon_max': '0'   
}
# Kernel function to check if particle is out of bounds and delete it
def BoundaryCheck(particle, fieldset, time):
    """Remove particle if it's out of the fieldset's latitude/longitude bounds."""
    if (
           particle.lon < fieldset.lon_min - 0.03
        or particle.lon > fieldset.lon_max - 0.03
        or particle.lat < fieldset.lat_min - 0.03
        or particle.lat > fieldset.lat_max - 0.03
    ):
        particle.delete()

# Define default extra variables for the TotalDistance kernel
TotalDistance_vars = [
    parcels.Variable("distance", initial=0.0, dtype=np.float32),
    parcels.Variable(
        "prev_lon", dtype=np.float32, to_write=False, initial=attrgetter("lon")
    ),
    parcels.Variable(
        "prev_lat", dtype=np.float32, to_write=False, initial=attrgetter("lat")
    ),
]

# Kernel function to calculate total distance
def TotalDistance(particle, fieldset, time):
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    lon_dist = (
        (particle.lon - particle.prev_lon)
        * 1.11e2
        * math.cos(particle.lat * math.pi / 180)
    )
    particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat
