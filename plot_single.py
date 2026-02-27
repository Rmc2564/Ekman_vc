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

print(Delta_Omega)
