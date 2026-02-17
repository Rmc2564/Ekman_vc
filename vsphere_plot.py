import numpy as np
import matplotlib.pyplot as plt
import h5py as hf

d_omega = 0.001

fig, ax = plt.subplots(1,3,figsize=(16,8),subplot_kw={'projection': 'polar'})
path = "./velocity/velocity_s5.h5"
data = hf.File(path)
u = data['tasks']['velocity']

u_arr = np.array(u)
print(np.shape(u_arr))

u_phi = u_arr[10,-1,:,0,0]
u_theta = u_arr[10,-1,0,:,0]
u_r = u_arr[10,-1,0,0,:]


print(np.shape(u_theta))