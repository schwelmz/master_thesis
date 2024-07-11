import numpy as np
import matplotlib.pyplot as plt
import settings

# read parameters
alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L = settings.read_parameters()
settings.print_settings()

xstart = 0
xend = 1
tstart = 0
tend = 0.1
Nx = 25
Nt = 7000
hx = (xend-xstart)/(Nx-1)
ht = (tend-tstart)/(Nt)
xs = np.linspace(xstart,xend,Nx)
alpha = D_N*ht/hx**2

def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    return zahler/nenner

#initial condition
N = np.zeros((Nt,Nx))
L = np.zeros((Nt,Nx))
N[0,10:12] = 10
L[0,10:12] = 10

fig, axs = plt.subplots(1,2)
# main loop
for n in range(0,Nt-1):
    print(n+1,"/",Nt)
    for i in range(1,Nx-1):
        dN_dxx = (N[n,i-1]-2*N[n,i]+N[n,i+1])/(hx**2)
        dL_dxx = (L[n,i-1]-2*L[n,i]+L[n,i+1])/(hx**2)
        f1 = alpha_N*hill_equation(N[n,i],L[n,i])
        f2 = alpha_L* hill_equation(N[n,i],L[n,i])
        N[n+1,i] = N[n,i] + ht*(f1 - gamma_N*N[n,i] + D_N*dN_dxx)
        L[n+1,i] = L[n,i] + ht*(f2 - gamma_L*L[n,i] + D_L*dL_dxx)
        # N[n+1,i] = N[n,i] + ht*(-gamma_N*N[n,i] + D_N*dN_dxx)
        # L[n+1,i] = L[n,i] + ht*(-gamma_L*L[n,i] + D_L*dL_dxx)
        # N[n+1,i] = N[n,i] + ht * dN_dxx
        # L[n+1,i] = L[n,i] + ht*(D_L*dL_dxx)
    plt.clf()
    plt.plot(xs, N[n+1,:], label="Nodal")
    plt.plot(xs, L[n+1,:], label="Lefty")
    plt.legend()
    plt.ylim(0,15)
    plt.pause(0.1)