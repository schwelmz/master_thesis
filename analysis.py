import numpy as np
import matplotlib.pyplot as plt
import settings
import visualization as vis

# read parameters
alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L = settings.read_parameters()
xstart = 0
xend = 5
ystart = 0
yend = 5
tstart = 0
tend = 10
ht = 5e-5
hx = 1.3e-1
hy = 1.3e-1
extent = [xstart,xend,ystart,yend]

N = np.load(f"out/Nodal_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy")
L = np.load(f"out/Lefty_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy")

# fig, axs = plt.subplots(1,2,figsize=(12,5))
# vis.heatmap(fig,axs, N, L, -3, extent,ht)
# plt.show()

vis.create_gif(N,L,extent,50000,ht,10)