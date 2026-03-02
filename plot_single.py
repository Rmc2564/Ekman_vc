import matplotlib.pyplot as plt
import matplotlib
import h5py

import numpy as np
from matplotlib import ticker, font_manager
import warnings
import os
warnings.filterwarnings("ignore")

import scipy.interpolate as inp
from control_parameters import parameters
locals().update(parameters)

def my_interp2d(f, rad, radnew):
    r = rad
    rnew = radnew
    fnew = np.zeros_like(f)
    for i in range(f.shape[0]):
        val = f[i, :]
        tckp = inp.splrep(r, val)
        fnew[i, :] = inp.splev(rnew, tckp)

    return fnew

def plot_stream(r,theta,vr_n,vtheta_n,density,label=None,clim=[0,0]):

    rad = np.linspace(r[-1], r[0], len(r))
    theta = np.linspace(0,np.pi,len(theta))

    rr, ttheta = np.meshgrid(rad, theta)
    plt.figure()
    fig,ax = plt.subplots(1,1,figsize=(6,6),subplot_kw={'projection': 'polar'})

    un = vr_n[:,::-1]
    vn = vtheta_n[:,::-1]/rr[:,::-1]

    un = my_interp2d(un, r[::-1], rad)
    vn = my_interp2d(vn, r[::-1], rad)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rorigin(0)
    ax.set_ylim(r.min(),r.max())
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    strm = ax.streamplot(ttheta.T,rr.T,vn.T,un.T,color='#d95f02',density=density,
                         broken_streamlines=True,linewidth=1)

    fig.tight_layout()

def coords_angular(path: str) -> np.ndarray | np.ndarray:
    data = h5py.File(path, mode='r')
    u_n_phi = data['tasks']['u_n_phi']
    r = u_n_phi.dims[3][0][:].ravel()
    theta = u_n_phi.dims[2][0][:].ravel()
    return r, theta

def get_angular(rs: np.ndarray, thetas: np.ndarray, u_phi: np.ndarray) -> np.ndarray:
    omega=np.zeros((len(thetas),len(rs)))
    for i in range(len(rs)):
        omega[:,i]=u_phi[:,i]/(rs[i]*np.sin(thetas)[:])
    return omega

def plot_angular(path: str, j: int, ax: matplotlib.projections.polar.PolarAxes) -> None:
    
    '''
    Takes an output of viscous_sphere.py and plots the angular velocity.

    :param path: Path to an AZ_avg_s*.h5 file.
    :param j: Integer used to select the time plotted.
    :param ax: Pre-defined matplotlib polar axis on which to plot the data.
    '''
    data = h5py.File(path, mode='r')
    u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]
    r, theta = coords_angular(path)
    
    omega=get_angular(r,theta,u_n_phi)
    r_m, theta_m = np.meshgrid(r,theta)

    time = np.array(data['scales/sim_time'])
    ax.pcolormesh(theta_m,r_m,omega,clim=(0,Delta_Omega),cmap='RdBu_r',edgecolors='face')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rorigin(0)
    ax.set_ylim(r.min(),r.max())
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r'$t =$'+str(time[j])[:4])

'''
Plots angular velocity at different times.
'''

fig,ax = plt.subplots(1,3,figsize=(16,8),subplot_kw={'projection': 'polar'})
j=10

path_1 = './AZ_avg/AZ_avg_s1.h5'
plot_angular(path_1,10,ax[0])
path_2 = './AZ_avg/AZ_avg_s3.h5'
plot_angular(path_2,10,ax[1])
path_3 ='./AZ_avg/AZ_avg_s6.h5'
plot_angular(path_3,10,ax[2])
plt.savefig("Angular_speeds.png")
plt.close()

j = 10

file_list = sorted(os.listdir('./AZ_avg'))

path_list = []
for file in file_list:
    print(file)
    path = "./AZ_avg/"+file
    path_list.append(path)

omega_rs = []
times = []
print(file_list)
for path in path_list:
    data = h5py.File(path, mode='r')
    time = np.array(data['scales/sim_time'])
    r, theta = coords_angular(path)
    u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]
    omega = get_angular(r, theta, u_n_phi)
    omega_r = omega[32][30]
    omega_rs.append(omega_r)
    times.append(time[j])
    

print(np.shape(omega_rs))
print(np.shape(times))

plt.plot(sorted(times),sorted(omega_rs),linewidth=2.0)
plt.show()