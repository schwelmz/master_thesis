import numpy as np
import matplotlib.pyplot as plt
import settings
import time
import visualization as vis
import os
import shutil

# read parameters
alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L, videomode, outdir = settings.read_parameters()
settings.print_settings()

xstart = 0
xend = 1000
ystart = 0
yend = 1000
tstart = 0
tend = 60*10
# hx = 10  #1.3e-1
# hy = 10  #1.3e-1
# ht = 1e-2   #5e-5
# Nx = int((xend-xstart)/hx + 1)
# Ny = int((yend-ystart)/hy + 1)
# Nt = int((tend-tstart)/ht + 1)
Nx = 500
Ny = 500
Nt = int(5e5)
hx = (xend-xstart)/(Nx-1)
hy = (xend-xstart)/(Ny-1)
ht = (tend-tstart)/(Nt-1)
xs = np.linspace(xstart,xend,Nx)
ys = np.linspace(ystart,yend,Ny)

print(f"hx={hx:.2e}, hy={hy:.2e}, ht={ht:.2e}")
print(f"Nx={Nx}, Ny={Ny}, Nt={Nt}")

def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    return zahler/nenner

def central_differences(U):
    return (U[0:-2,1:-1] + U[2:,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] - 4*U[1:-1,1:-1])/(hx**2)

def initialize_matrix(rows, cols, option="white-noise"):
    init_val = 1
    percentage = 0.01
    if option == "4dots":
        array = np.zeros((rows,cols))
        d_x = cols//4
        d_y = rows//4
        array[d_x,d_y] = init_val
        array[3*d_x,3*d_y] = init_val
        array[d_x,3*d_y] = init_val
        array[3*d_x,d_y] = init_val
        return array
    elif option == "random-dots":
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
    elif option == "white-noise":
        # Generate a 2D array with white noise using a normal distribution
        # low = 0.0
        # high = 1.0
        # return np.random.uniform(low, high, (rows, cols))u-=1
        return np.random.rand(rows,cols)

#initial condition
N_init = initialize_matrix(Nx,Ny, option="white-noise")
L_init = initialize_matrix(Nx,Ny, option="white-noise")

#set up image saving
frames = 250
frameskips = Nt//frames

#create output directory
if not os.path.exists(f"{outdir}/plots") or os.path.exists(f"{outdir}/data"):
    os.makedirs(f"{outdir}/plots")
    os.makedirs(f"{outdir}/data")

#main loop
tik = time.time()
N_new = np.zeros((Nx,Ny))
L_new = np.zeros((Nx,Ny))
N_old = N_init
L_old = L_init
for n in range(0,Nt-1):
    print(f"\rtime step {n+1}/{Nt}",end=" ",flush=True)
    #approximate the Laplacian operator using central differences
    Lap_N = central_differences(N_old)
    Lap_L = central_differences(L_old)
    #calculate the hill equation term
    hill_term = hill_equation(N_old[1:-1,1:-1],L_old[1:-1,1:-1])
    #calculate Nodal and Lefty at the new time step
    N_new[1:-1,1:-1] = N_old[1:-1,1:-1] + ht*(alpha_N*hill_term - gamma_N*N_old[1:-1,1:-1] + D_N*Lap_N)
    L_new[1:-1,1:-1] = L_old[1:-1,1:-1] + ht*(alpha_L*hill_term - gamma_L*L_old[1:-1,1:-1] + D_L*Lap_L)
    #set Neumann boundary values
    N_new[0,:] = N_new[1,:]     #left
    L_new[0,:] = L_new[1,:]
    N_new[:,0] = N_new[:,1]     #bottom
    L_new[:,0] = L_new[:,1]
    N_new[-1,:] = N_new[-2,:]   #right
    L_new[-1,:] = L_new[-2,:]
    N_new[:,-1] = N_new[:,-2]   #top
    L_new[:,-1] = L_new[:,-2]

    #save plots
    if videomode:
        if n%frameskips == 0 or n in[1,2,3,4,5,10,40,80,150]:
            fig, axs = plt.subplots(1,2,figsize=(12,5))
            img = vis.heatmap(fig,axs,N_new,L_new,n,[xstart,xend,ystart,yend],ht)
            fig.savefig(f"{outdir}/plots/heatmap_{n}")
            plt.close()

    #set new state as old
    N_old = N_new
    L_old = L_new
    
#print computation time
tok = time.time()
print(f"\ndone! time taken: {(tok-tik)/60:.1f}min")

#save data of last time step
np.save(f"{outdir}/data/Nodal_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy",N_new)
np.save(f"{outdir}/data/Lefty_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy",L_new)