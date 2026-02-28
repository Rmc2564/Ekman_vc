import matplotlib.pyplot as plt
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

fig,ax = plt.subplots(1,3,figsize=(16,8),subplot_kw={'projection': 'polar'})

dire = './AZ_avg/AZ_avg_s1.h5'
data = h5py.File(dire, mode='r')
u_n_phi = data['tasks']['u_n_phi']
print(np.shape(u_n_phi))    
time = np.array(data['scales/sim_time'])  
theta = u_n_phi.dims[2][0][:].ravel()
r = u_n_phi.dims[3][0][:].ravel()

j = 10

u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]

#Convert v_phi to an Angular velocity
omega=np.zeros((len(theta),len(r)))
for i in range(len(r)):
    omega[:,i]=u_n_phi[:,i]/(r[i]*np.sin(theta)[:])
r_m, theta_m = np.meshgrid(r,theta)

ax[0].pcolormesh(theta_m,r_m,u_n_phi,clim=(0,Delta_Omega),cmap='bone_r',edgecolors='face')
ax[0].set_theta_zero_location('N')
ax[0].set_theta_direction(-1)
ax[0].set_rorigin(0)
ax[0].set_ylim(r.min(),r.max())
ax[0].set_thetamin(0)
ax[0].set_thetamax(180)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(r'$t =$'+str(time[j])[:4])


dire = './AZ_avg/AZ_avg_s3.h5'
data = h5py.File(dire, mode='r')
time = np.array(data['scales/sim_time']) 
u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]

#Convert v_phi to an Angular velocity
omega=np.zeros((len(theta),len(r)))
for i in range(len(r)):
    omega[:,i]=u_n_phi[:,i]/(r[i]*np.sin(theta)[:])
r_m, theta_m = np.meshgrid(r,theta)

ax[1].pcolormesh(theta_m,r_m,u_n_phi,clim=(0,Delta_Omega),cmap='bone_r',edgecolors='face')
ax[1].set_theta_zero_location('N')
ax[1].set_theta_direction(-1)
ax[1].set_rorigin(0)
ax[1].set_ylim(r.min(),r.max())
ax[1].set_thetamin(0)
ax[1].set_thetamax(180)
ax[1].grid(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title(r'$t =$'+str(time[j])[:4])

dire = './AZ_avg/AZ_avg_s5.h5'
data = h5py.File(dire, mode='r')
time = np.array(data['scales/sim_time']) 
u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]

#Convert v_phi to an Angular velocity
omega=np.zeros((len(theta),len(r)))
for i in range(len(r)):
    omega[:,i]=u_n_phi[:,i]/(r[i]*np.sin(theta)[:])
r_m, theta_m = np.meshgrid(r,theta)

ax[2].pcolormesh(theta_m,r_m,u_n_phi,clim=(0,Delta_Omega),cmap='bone_r',edgecolors='face')
ax[2].set_theta_zero_location('N')
ax[2].set_theta_direction(-1)
ax[2].set_rorigin(0)
ax[2].set_ylim(r.min(),r.max())
ax[2].set_thetamin(0)
ax[2].set_thetamax(180)
ax[2].grid(False)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title(r'$t =$'+str(time[j])[:4])

#plt.savefig("Velocity_early.png")

fig,ax = plt.subplots(1,3,figsize=(16,8),subplot_kw={'projection': 'polar'})

dire = './AZ_avg/AZ_avg_s6.h5'
data = h5py.File(dire, mode='r')
u_n_phi = data['tasks']['u_n_phi']    
time = np.array(data['scales/sim_time'])  
theta = u_n_phi.dims[2][0][:].ravel()
r = u_n_phi.dims[3][0][:].ravel()

j = 10

u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]

#Convert v_phi to an Angular velocity
omega=np.zeros((len(theta),len(r)))
for i in range(len(r)):
    omega[:,i]=u_n_phi[:,i]/(r[i]*np.sin(theta)[:])
r_m, theta_m = np.meshgrid(r,theta)

ax[0].pcolormesh(theta_m,r_m,u_n_phi,clim=(0,Delta_Omega),cmap='bone_r',edgecolors='face')
ax[0].set_theta_zero_location('N')
ax[0].set_theta_direction(-1)
ax[0].set_rorigin(0)
ax[0].set_ylim(r.min(),r.max())
ax[0].set_thetamin(0)
ax[0].set_thetamax(180)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(r'$t =$'+str(time[j])[:4])


dire = './AZ_avg/AZ_avg_s5.h5'
data = h5py.File(dire, mode='r')
time = np.array(data['scales/sim_time']) 
u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]

#Convert v_phi to an Angular velocity
omega=np.zeros((len(theta),len(r)))
for i in range(len(r)):
    omega[:,i]=u_n_phi[:,i]/(r[i]*np.sin(theta)[:])
r_m, theta_m = np.meshgrid(r,theta)

ax[1].pcolormesh(theta_m,r_m,u_n_phi,clim=(0,Delta_Omega),cmap='bone_r',edgecolors='face')
ax[1].set_theta_zero_location('N')
ax[1].set_theta_direction(-1)
ax[1].set_rorigin(0)
ax[1].set_ylim(r.min(),r.max())
ax[1].set_thetamin(0)
ax[1].set_thetamax(180)
ax[1].grid(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title(r'$t =$'+str(time[j])[:4])

dire = './AZ_avg/AZ_avg_s6.h5'
data = h5py.File(dire, mode='r')
time = np.array(data['scales/sim_time']) 
u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]

#Convert v_phi to an Angular velocity
omega=np.zeros((len(theta),len(r)))
for i in range(len(r)):
    omega[:,i]=u_n_phi[:,i]/(r[i]*np.sin(theta)[:])
r_m, theta_m = np.meshgrid(r,theta)

ax[2].pcolormesh(theta_m,r_m,u_n_phi,clim=(0,Delta_Omega),cmap='bone_r',edgecolors='face')
ax[2].set_theta_zero_location('N')
ax[2].set_theta_direction(-1)
ax[2].set_rorigin(0)
ax[2].set_ylim(r.min(),r.max())
ax[2].set_thetamin(0)
ax[2].set_thetamax(180)
ax[2].grid(False)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title(r'$t =$'+str(time[j])[:4])

#plt.savefig("velocity_late.png")

plt.show()
