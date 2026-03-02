import matplotlib.pyplot as plt
import matplotlib
import h5py

import numpy as np
from matplotlib import ticker, font_manager
import warnings
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

def angular_coords(dire: str) -> np.ndarray | np.ndarray:
    '''
    Returns coordinates from a spin up simulation using dedalus.

    :param dire: File path to h5 file containing the data.
    :returns r: radial coordinates.
    :returns theta: Polar angle.
    '''

    data = h5py.File(dire, mode='r')
    u_n_phi = data['tasks']['u_n_phi']
    r = u_n_phi.dims[3][0][:].ravel()
    theta = u_n_phi.dims[2][0][:].ravel()
    return r, theta

def plot_angular(dire: str, j: int, ax: matplotlib.projections.polar.PolarAxes) -> None:
    
    '''
    Takes an output of viscous_sphere.py and plots the angular velocity.

    :param dire: Path to an AZ_avg_s*.h5 file.
    :param j: Integer used to select the time plotted.
    :param ax: Pre-defined polar matplotlib axis on which to plot the data.
    '''

    data = h5py.File(dire, mode='r')
    u_n_phi = data['tasks']['u_n_phi']
    r = u_n_phi.dims[3][0][:].ravel()
    theta = u_n_phi.dims[2][0][:].ravel()
    u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]
    

    omega=np.zeros((len(theta),len(r)))
    for i in range(len(r)):
        omega[:,i]=u_n_phi[:,i]/(r[i]*np.sin(theta)[:])
    r_m, theta_m = np.meshgrid(r,theta)

    r_m, theta_m=np.meshgrid(r,theta)
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

dire_1 = './AZ_avg/AZ_avg_s1.h5'
plot_angular(dire_1,10,ax[0])

dire_2 = './AZ_avg/AZ_avg_s3.h5'
plot_angular(dire_2,10,ax[1])
dire_3 ='./AZ_avg/AZ_avg_s6.h5'
plot_angular(dire_3,10,ax[2])
plt.show()

