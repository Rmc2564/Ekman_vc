import sys
import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)


# Parameters - load in from parameter file

from control_parameters import parameters
locals().update(parameters)

# Additional Parameters
radius = 1
timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)

if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))


# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=1, dealias=dealias, dtype=dtype)
sphere = ball.surface

#Fields
u_n = dist.VectorField(coords, name='u_n',bases=ball)
p_n = dist.Field(name='p_n', bases=ball)
tau_p_n = dist.Field(name='tau_p_n')
tau_u_n = dist.VectorField(coords, name='tau_u_n', bases=sphere)

#Substitutions
phi, theta, r = dist.local_grids(ball)
er = dist.VectorField(coords)
etheta = dist.VectorField(coords)
ephi = dist.VectorField(coords)
er['g'][2] = 1
etheta['g'][1] = 1
ephi['g'][0] = 1

ez = dist.VectorField(coords, bases=ball)
ez['g'][1] = -np.sin(theta)
ez['g'][2] = np.cos(theta) # unit vector in z direction

sintheta = dist.Field(name='sintheta', bases=ball)
sintheta['g'] = np.sin(theta)

lift = lambda A: d3.Lift(A, ball, -1)

dot = d3.DotProduct
curl = d3.Curl
cross = d3.CrossProduct

# Problem
problem = d3.IVP([p_n, u_n, tau_p_n, tau_u_n], namespace=locals())
problem.add_equation("div(u_n) + tau_p_n = 0")
problem.add_equation("dt(u_n) + grad(p_n) - Ek*lap(u_n) + lift(tau_u_n)  = -u_n@grad(u_n) - 2*cross(omega_n,u_n) - cross(curl(u_n),u_n) - cross(curl(u_n),cross(curl(u_n),r_vec))")
problem.add_equation("angular(u_n(r=radius)) = angular(uang_R1)") # spin up at outer boundary
problem.add_equation("radial(u_n(r=radius)) = 0") # impenetrable bc
problem.add_equation("integ(p_n) = 0")  # Pressure gauge normal fluid

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#Initial condition - solid body rotation
for i in range(0,len(r)):
    u_n[0][:,i] = Omega_Init*r[i]*np.sin(theta)[:]
