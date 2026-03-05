import sys
import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

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
rsintheta = dist.Field(name='r_field',bases = ball)
rsintheta['g'] = r*np.sin(theta)

lift = lambda A: d3.Lift(A, ball, -1)

dot = d3.DotProduct
curl = d3.Curl
cross = d3.CrossProduct

Omega_R = Delta_Omega + Omega_Init

uang_R1 = dist.VectorField(coords, bases=ball)(r=radius).evaluate()
uang_R1['g'][0,:] = (Omega_R*sintheta)(r=radius).evaluate()['g']

# Problem
problem = d3.IVP([p_n, u_n, tau_p_n, tau_u_n], namespace=locals())
problem.add_equation("div(u_n) + tau_p_n = 0")
problem.add_equation("dt(u_n) + grad(p_n) - Ek*lap(u_n) + lift(tau_u_n) = - u_n@grad(u_n)")
problem.add_equation("angular(u_n(r=radius)) = angular(uang_R1)") # spin up at outer boundary
problem.add_equation("radial(u_n(r=radius)) = 0") # impenetrable bc
problem.add_equation("integ(p_n) = 0")  # Pressure gauge normal fluid

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#Initial condition - solid body rotation

u_n['g'][0] = r*np.sin(theta) 

#Analysis

volume = (4/3)*np.pi*radius**3
az_avg = lambda A: d3.Average(A, coords.coords[0])
s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
vol_avg = lambda A: d3.Integrate(A/volume, coords)

# define every component of velocity (for output)
u_n_r = dot(u_n,er)
u_n_theta = dot(u_n,etheta)
u_n_phi = dot(u_n, ephi)

AZ_avg = solver.evaluator.add_file_handler('AZ_avg', sim_dt=0.05, max_writes=100)
AZ_avg.add_task(az_avg(dot(er,u_n)), name='u_n_r')
AZ_avg.add_task(dot(etheta,u_n), name='u_n_theta')
AZ_avg.add_task(az_avg(u_n_phi), name='u_n_phi')


slices = solver.evaluator.add_file_handler('slices', sim_dt=0.025, max_writes=100)

slices.add_task(u_n_phi(theta=np.pi/2), scales=dealias, name='u_n_phi(equator)')

# Checkpoint
checkpoint = solver.evaluator.add_file_handler('checkpoint', wall_dt=3600, max_writes=1, parallel='gather')
checkpoint.add_tasks(solver.state, layout='g')

# CFL
CFL = d3.CFL(solver, max_timestep, cadence=1, safety=0.3, threshold=0.1, max_dt=max_timestep)
CFL.add_velocity(u_n)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_n@u_n)*Ek, name='Re_n')

# Main loop
solver.evolve(timestep_function=CFL.compute_timestep, log_cadence = 10)
