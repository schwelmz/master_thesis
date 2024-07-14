import numpy as np
import matplotlib.pyplot as plt
import settings
import time
import visualization as vis

# read parameters
alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L = settings.read_parameters()
settings.print_settings()

xstart = 0
xend = 15
ystart = 0
yend = 15
tstart = 0
tend = 10
hx = 1.3e-1
hy = 1.3e-1
ht = 5e-5
Nx = int((xend-xstart)/hx + 1)
Ny = int((yend-ystart)/hy + 1)
Nt = int((tend-tstart)/ht + 1)
xs = np.linspace(xstart,xend,Nx)
ys = np.linspace(ystart,yend,Ny)
N = np.zeros((Nt,Nx,Ny))
L = np.zeros((Nt,Nx,Ny))

print(f"hx={hx:.2e}, hy={hy:.2e}, ht={ht:.2e}")

def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    return zahler/nenner

def initialize_matrix(rows, cols, init_val=100, percentage=0.05):
    # Step 1: Create a 2D array filled with zeros
    array = np.zeros((rows, cols), dtype=int)
    
    # Step 2: Calculate the number of ones needed
    total_elements = rows * cols
    num_ones = int(total_elements * percentage)
    
    # Step 3: Randomly select positions to place the ones
    ones_positions = np.random.choice(total_elements, num_ones, replace=False)
    
    # Step 4: Convert 1D positions to 2D indices and place the ones
    for pos in ones_positions:
        row_index = pos // cols
        col_index = pos % cols
        array[row_index, col_index] = init_val
    
    return array

#initial condition
N[0,:,:] = initialize_matrix(Nx,Ny)
L[0,:,:] = initialize_matrix(Nx,Ny)
# m = Nx//2
# N[0,m,m] = 1
# L[0,m,m] = 1

fig, axs = plt.subplots(1,2,figsize=(12,5))
#main loop
tik = time.time()
for n in range(0,Nt-1):
    print(f"\rtime step {n+1}/{Nt}",end=" ",flush=True)
    for i in range(0,Nx):
        for j in range(0,Ny):
            #calculate the diffusion term using 2nd order finite differneces (with periodic BC)
            dN_dxx = (N[n,(i-1)%Nx,j]-2*N[n,i,j]+N[n,(i+1)%Nx,j])/(hx**2)
            dN_dyy = (N[n,i,(j-1)%Ny]-2*N[n,i,j]+N[n,i,(j+1)%Ny])/(hy**2)
            dL_dxx = (L[n,(i-1)%Nx,j]-2*L[n,i,j]+L[n,(i+1)%Nx,j])/(hx**2)
            dL_dyy = (L[n,i,(j-1)%Ny]-2*L[n,i,j]+L[n,i,(j+1)%Ny])/(hy**2)
            #calculate the hill equation term
            f1 = alpha_N*hill_equation(N[n,i,j],L[n,i,j])
            f2 = alpha_L* hill_equation(N[n,i,j],L[n,i,j])
            #calculate Nodal and Lefty at the new time step
            N[n+1,i,j] = N[n,i,j] + ht*(f1 - gamma_N*N[n,i,j] + D_N*(dN_dxx+dN_dyy))
            L[n+1,i,j] = L[n,i,j] + ht*(f2 - gamma_L*L[n,i,j] + D_L*(dL_dxx+dL_dyy))
    if n%1000==0:
        np.save(f"out/Nodal_quicksave_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy",N)
        np.save(f"out/Lefty_quicksave_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy",L)
    #liveplot
    if False:
        tok = time.time()
        print(f"\ndone! time taken: {(tok-tik)/60:.1f}min")
        vis.heatmap(fig,axs,N,L,n,[xstart,xend,ystart,yend],ht)
        plt.pause(0.5)
        cb0.remove()
        cb1.remove()
        axs[0].cla()
        axs[1].cla()

#save data
np.save(f"out/Nodal_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy",N)
np.save(f"out/Lefty_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy",L)