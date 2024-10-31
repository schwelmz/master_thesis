import numpy as np
import matplotlib.pyplot as plt
import settings
import visualization as vis

# read parameters
# parameters = settings.read_parameters()
# settings.print_settings(parameters)

# #read command line arguments
# args = settings.read_cmdline_args()
# videomode = args.videomode
# outdir = args.outdir
# setup = args.model

# #specify setup
# if setup == "NL":   #Nodal-Lefty
#     alpha_N = float(parameters['alpha_N'])
#     alpha_L = float(parameters['alpha_L'])
#     k_p = float(parameters['k_p'])
#     n_N = float(parameters['n_N'])
#     n_L = float(parameters['n_L'])
#     K_N = float(parameters['K_N'])
#     K_L = float(parameters['K_L'])
#     gamma_N = float(parameters['gamma_N'])
#     gamma_L = float(parameters['gamma_L'])
#     D_N = float(parameters['D_N'])
#     D_L = float(parameters['D_L'])
#     xstart = 0
#     xend = 1000
#     ystart = 0
#     yend = 1000
#     tstart = 0
#     tend = 60*100
#     Nx = 100
#     Ny = 100
#     Nt = int(5e4)
# elif setup == "GM":     #Gierer-Meinhardt
#     D_u = float(parameters["D_u"])
#     D_v = float(parameters["D_v"])
#     mu = float(parameters["mu"])
#     a = float(parameters["a"])
#     c = float(parameters["c"])
#     r = float(parameters["r"])
#     xstart = 0
#     xend = 100
#     ystart = 0
#     yend = 100
#     tstart = 0
#     tend = 100
#     Nx = 200
#     Ny = 200
#     Nt = int(1e4)

# #Define the spatial and temporal grid
# hx = (xend-xstart)/(Nx-1)
# hy = (xend-xstart)/(Ny-1)
# ht = (tend-tstart)/(Nt-1)
# xs = np.linspace(xstart,xend,Nx)
# ys = np.linspace(ystart,yend,Ny)
# print(f"hx={hx:.2e}, hy={hy:.2e}, ht={ht:.2e}")
# print(f"Nx={Nx}, Ny={Ny}, Nt={Nt}")

# a = 0.1
# b = 1
# nullcline_f = lambda u: (u**2)/(b*u-a)
# nullcline_g = lambda u: u**2
# us = np.linspace(0.1,3,1000)
# plt.plot(us, nullcline_f(us), label='f(u,v)=0')
# plt.plot(us, nullcline_g(us), label='g(u,v)=0')
# plt.legend()
# plt.show()

#import data
# N = np.load(f"out/neumann/Nodal_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}_{n_N}_{n_L}.npy")
# L = np.load(f"out/neumann/Lefty_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}_{n_N}_{n_L}.npy")
# N = np.load(f"out/neumann/Nodal_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy")
# L = np.load(f"out/neumann/Lefty_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy")

#plot last state
# fig, axs = plt.subplots(1,2,figsize=(12,5))
# vis.heatmap(fig,axs, N[20000,:,:], L[20000,:,:], 20000, extent,ht)
# plt.show()

# create gif
# Nt = N.shape[0]
# vis.create_gif(N,L,extent,Nt,ht,100)

parameters = settings.read_parameters()
settings.print_settings(parameters)
alpha_N = float(parameters['alpha_N'])
alpha_L = float(parameters['alpha_L'])
n_N = float(parameters['n_N'])
n_L = float(parameters['n_L'])
K_N = float(parameters['K_N'])
K_L = float(parameters['K_L'])
gamma_N = float(parameters['gamma_N'])
gamma_L = float(parameters['gamma_L'])
D_N = float(parameters['D_N'])
D_L = float(parameters['D_L'])

# alpha_N = 181.82
# alpha_L = 112.82
# gamma = 2.384
# n_N = 2.63
# n_L = 1.09

#dimensionless parameters:
alpha_N_ = alpha_N/(gamma_N*K_N)
alpha_L_ = alpha_L/(gamma_N*K_L)
gamma_ = gamma_L/gamma_N
d = D_L/D_N
print("alpha_N_ = ",alpha_N_)
print("alpha_L_ = ",alpha_L_)
print("gamma_ = ", gamma_)
print("d = ",d)

nullcline_f_2 = lambda u: ((alpha_N_*u**(n_N-1) - u**n_N)**(1/n_N) - 1)**(1/n_L)
nullcline_g_2 = lambda v: ((gamma_*v)/(alpha_L_-gamma_*v))**(1/n_N)*(1+v**n_L)

us = np.linspace(0,1000,10000)
vs = np.linspace(0,1000,10000)
plt.plot(us,nullcline_f_2(us),label="f(u,v)=0")
plt.plot(nullcline_g_2(vs),vs,label="g(u,v)=0")
plt.xlabel("u")
plt.ylabel("v")
# plt.ylim(0,0.1)
plt.legend()
plt.show()