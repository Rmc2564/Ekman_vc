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

#Field
u = dist.VectorField(coords, name='u',bases = ball)
p = dist.Field(name = 'p', bases = ball)
tau_p = dist.Field(name="tau_p")
tau_u = dist.VectorField(coords, name="tau_u", bases=sphere)

#Substitutions
phi, theta, r = dist.local_grids(ball)
Ek = PARAMS['Ek']
lift = lambda A: d3.Lift(A, ball, -1)

ang_boundary = dist.VectorField(coords, name = 'spup', bases=sphere)
ang_boundary['g'][0,:] = PARAMS['radius']*np.sin(theta)

#Problem
problem = d3.IVP([u, p, tau_p, tau_u], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - Ek*lap(u) + lift(tau_u) = -cross(curl(u),u)")
problem.add_equation("radial(u(r=PARAMS['radius'])) = 0") #Impenetrable boundary condition
problem.add_equation("angular(u(r=PARAMS['radius'])) = angular(ang_boundary)")
problem.add_equation("integ(p) = 0") #Pressure gauge

#Solver
solver = problem.build_solver(PARAMS['timestepper'])
solver.stop_time = PARAMS["stop_time"]
