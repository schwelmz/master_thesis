import numpy as np
import matplotlib.pyplot as plt
import settings
import visualization as vis

# read parameters
alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L = settings.read_parameters()
xstart = 0
xend = 20
ystart = 0
yend = 20
tstart = 0
tend = 1
ht = 5e-5
hx = 1.3e-1
hy = 1.3e-1
extent = [xstart,xend,ystart,yend]

#import data
N = np.load(f"out/Nodal_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}_0.01.npy")
L = np.load(f"out/Lefty_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy")

#plot last state
fig, axs = plt.subplots(1,2,figsize=(12,5))
vis.heatmap(fig,axs, N[-1,:,:], L[-1,:,:], -1, extent,ht)
plt.show()

# create gif
# Nt = N.shape[0]
# vis.create_gif(N,L,extent,Nt,ht,10)