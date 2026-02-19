import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
import csv
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

theta_save = list(theta[0,:,0])
r_save = list(r[0,0,:])
#Save theta and r
rows = [theta_save, r_save]

print(rows)

np.savetxt("Theta.csv", theta_save, delimiter=', ')
np.savetxt("r.csv", r_save, delimiter=', ')

#Spherical unit vectors
er = dist.VectorField(coords)
etheta = dist.VectorField(coords)
ephi = dist.VectorField(coords)
er['g'][2] = 1
etheta['g'][1] = 1
ephi['g'][0] = 1

u_r = d3.dot(u,er)
u_theta = d3.dot(u,etheta)
u_phi = d3.dot(u,ephi)

#Problem
problem = d3.IVP([u, p, tau_p, tau_u], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - Ek*lap(u) + lift(tau_u) = -cross(curl(u),u) - u@grad(u)")
problem.add_equation("radial(u(r=PARAMS['radius'])) = 0") #Impenetrable boundary condition
problem.add_equation("angular(u(r=PARAMS['radius'])) = angular(ang_boundary)")
problem.add_equation("integ(p) = 0") #Pressure gauge

#Solver
solver = problem.build_solver(PARAMS['timestepper'])
solver.stop_time = PARAMS["stop_time"]

#Initial conditions
u.fill_random('g', seed=42, distribution='normal', scale=1e-10)
u.low_pass_filter(scales=0.5)
timestep = PARAMS['max_timestep']

print(u_phi['g'] == u['g'][0])
print(coords.coords)
quit()

#Analysis
vel = solver.evaluator.add_file_handler('velocity',sim_dt=0.025,max_writes=100)
vel.add_task(d3.Average(u_phi,coord=coords.coords[0]), name = 'u_phi')

#CFL
CFL = d3.CFL(solver, timestep, cadence=1, safety=0.3,
              threshold=0.1, max_dt=PARAMS["max_timestep"])
CFL.add_velocity(u)

#Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)*Ek, name='Re_n')

#Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re_n = flow.max('Re_n')
            logger.info("Iteration=%i, Time=%e, dt=%e, max(Re_n)=%e" 
                        %(solver.iteration, solver.sim_time, timestep, max_Re_n))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
