import numpy as np
import matplotlib.pyplot as plt
import h5py as hf

d_omega = 0.001

rs = np.loadtxt("r.csv")
thetas = np.loadtxt("Theta.csv")
print(rs)