import math
import parcels
import numpy as np
from   operator import attrgetter
from   fieldset_vars import FieldsetVariable

# boundary_check

boundary_check_vars = {
    'lat_min': FieldsetVariable('lat_min', lambda fieldset: fieldset.U.grid.lat[0]),
    'lat_max': FieldsetVariable('lat_max', lambda fieldset: fieldset.U.grid.lat[-1]),
    'lon_min': FieldsetVariable('lon_min', lambda fieldset: fieldset.U.grid.lon[0]),
    'lon_max': FieldsetVariable('lon_max', lambda fieldset: fieldset.U.grid.lon[-1]),
}

def boundary_check(particle, fieldset, time):
    """Remove particle if it's out of the fieldset's latitude/longitude bounds."""
    if (
           particle.lon < fieldset.lon_min - 0.03
        or particle.lon > fieldset.lon_max - 0.03
        or particle.lat < fieldset.lat_min - 0.03
        or particle.lat > fieldset.lat_max - 0.03
    ):
        particle.out_of_bounds = True
        particle.delete()

#pathlength

pathlength_vars = [
    parcels.Variable("distance", initial=0.0, dtype=np.float32),
    parcels.Variable(
        "prev_lon", dtype=np.float32, to_write=False, initial=attrgetter("lon")
    ),
    parcels.Variable(
        "prev_lat", dtype=np.float32, to_write=False, initial=attrgetter("lat")
    ),
    parcels.Variable("out_of_bounds", initial=False, dtype=np.bool_)
]

def pathlength(particle, fieldset, time):
    """Computes the total path length traveled by a particle."""
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    lon_dist = (
        (particle.lon - particle.prev_lon)
        * 1.11e2
        * math.cos(particle.lat * math.pi / 180)
    )
    particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat
