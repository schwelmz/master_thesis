import numpy as np
import matplotlib.pyplot as plt
import settings
import time
import visualization as vis

# read parameters
alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L = settings.read_parameters()
settings.print_settings()

xstart = 0
xend = 10
ystart = 0
yend = 10
tstart = 0
tend = 2
hx = 1.3e-1
hy = 1.3e-1
ht = 5e-5
Nx = int((xend-xstart)/hx + 1)
Ny = int((yend-ystart)/hy + 1)
Nt = int((tend-tstart)/ht + 1)
xs = np.linspace(xstart,xend,Nx)
ys = np.linspace(ystart,yend,Ny)

print(f"hx={hx:.2e}, hy={hy:.2e}, ht={ht:.2e}")

def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    return zahler/nenner

def initialize_random_matrix(rows, cols, init_val=100, percentage=0.01):
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

def initialize_matrix(rows, cols, init_val=1000):
    array = np.zeros((rows,cols))
    d_x = cols//4
    d_y = rows//4
    array[d_x,d_y] = init_val
    array[3*d_x,3*d_y] = init_val
    array[d_x,3*d_y] = init_val
    array[3*d_x,d_y] = init_val
    return array


#initial condition
N_init = initialize_matrix(Nx,Ny)
L_init = initialize_matrix(Nx,Ny)
# m = Nx//2
# N[0,m,m] = 1
# L[0,m,m] = 1

fig, axs = plt.subplots(1,2,figsize=(12,5))
#main loop
tik = time.time()
N_new = np.zeros((Nx,Ny))
L_new = np.zeros((Nx,Ny))
N_old = N_init
L_old = L_init
skips = 10
N_array = np.zeros((Nt,Nx,Ny))
L_array = np.zeros((Nt,Nx,Ny))
for n in range(0,Nt-1):
    print(f"\rtime step {n+1}/{Nt}",end=" ",flush=True)
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            #calculate the diffusion term using 2nd order finite differneces (with periodic BC)
            dN_dxx = (N_old[i-1,j]-2*N_old[i,j]+N_old[i+1,j])/(hx**2)
            dN_dyy = (N_old[i,j-1]-2*N_old[i,j]+N_old[i,j+1])/(hy**2)
            dL_dxx = (L_old[i-1,j]-2*L_old[i,j]+L_old[i+1,j])/(hx**2)
            dL_dyy = (L_old[i,j-1]-2*L_old[i,j]+L_old[i,j+1])/(hy**2)
            #calculate the hill equation term
            hill_term = hill_equation(N_old[i,j],L_old[i,j])
            #calculate Nodal and Lefty at the new time step
            N_new[i,j] = N_old[i,j] + ht*(alpha_N*hill_term - gamma_N*N_old[i,j] + D_N*(dN_dxx+dN_dyy))
            L_new[i,j] = L_old[i,j] + ht*(alpha_L*hill_term - gamma_L*L_old[i,j] + D_L*(dL_dxx+dL_dyy))
            #set Neumann boundary values
            N_new[0,:] = N_new[1,:]     #left
            L_new[0,:] = L_new[1,:]
            N_new[:,0] = N_new[:,1]     #bottom
            L_new[:,0] = L_new[:,1]
            N_new[-1,:] = N_new[-2,:]   #right
            L_new[-1,:] = L_new[-2,:]
            N_new[:,-1] = N_new[:,-2]   #top
            L_new[:,-1] = L_new[:,-2]

    #liveplot
    if False:
        cb0, cb1 = vis.heatmap(fig,axs,N_new,L_new,n,[xstart,xend,ystart,yend],ht)
        plt.pause(0.0001)
        cb0.remove()
        cb1.remove()
        axs[0].cla()
        axs[1].cla()
    #save solution
    N_array[n,:,:] = N_new
    L_array[n,:,:] = L_new
    #set new state as old
    N_old = N_new
    L_old = L_new
    
#print computation time
tok = time.time()
print(f"\ndone! time taken: {(tok-tik)/60:.1f}min")

#plot last time step
vis.heatmap(fig,axs,N_new,L_new,-1,[xstart,xend,ystart,yend],ht)
plt.show()

#save data
np.save(f"out/neumann/Nodal_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy",N_array)
np.save(f"out/neumann/Lefty_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}_{alpha_N}_{alpha_L}.npy",L_array)