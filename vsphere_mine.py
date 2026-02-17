import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)

#Parameters

PARAMS = {
    "Nphi": 128,
    "Ntheta": 64,
    "Nr": 128,
    "Omega": 1,
    "d_omega": 1e-3,
    "Ek": 5e-3,
    "dealias": 1.5,
    "stop_time": 15,
    "radius": 1, 
    "timestepper": d3.SBDF3,
    "CFL_safety": 0.2,
    "max_timestep": 1e-2,
    "dtype": np.float64
}
#MPI from original viscous_sphere.py
ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

#Coords
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=PARAMS['dtype'], mesh=mesh)
ball = d3.BallBasis(coords, shape=(PARAMS['Nphi'], PARAMS['Ntheta'], PARAMS['Nr']),
            radius=PARAMS["radius"], dealias=PARAMS['dealias'], dtype=PARAMS['dtype'])
sphere = ball.surface
