import numpy as np
import matplotlib.pyplot as plt
import settings

# read parameters
alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L = settings.read_parameters()
settings.print_settings()

xstart = 0
xend = 1
ystart = 0
yend = 1
tstart = 0
tend = 1
ht = 1e-2
hx = 1e-1
hy = 1e-1
Nx = int((xend-xstart)/hx)
Ny = int((yend-ystart)/hy)
Nt = int((tend-tstart)/ht)
xs = np.linspace(xstart,xend,Nx)
ys = np.linspace(ystart,yend,Ny)
N = np.zeros((Nt,Nx,Ny))
L = np.zeros((Nt,Nx,Ny))

def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    # if zahler != 0:
    #     print("true")
    return zahler/nenner

def initialize_matrix(rows, cols, init_val=1, percentage=0.05):
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
# N[0,:,:] = initialize_matrix(Nx,Ny)
# L[0,:,:] = initialize_matrix(Nx,Ny)
m = Nx//2
N[0,m-1:m+1,m-1:m+1] = 0.01
L[0,m-1:m+1,m-1:m+1] = 0.01

fig, axs = plt.subplots(1,2)
#main loop
for n in range(0,Nt-1):
    print(n+1,"/",Nt)
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            dN_dxx = (N[n,i-1,j]-2*N[n,i,j]+N[n,i+1,j])/(hx**2)
            dN_dyy = (N[n,i,j-1]-2*N[n,i,j]+N[n,i,j+1])/(hy**2)
            dL_dxx = (L[n,i-1,j]-2*L[n,i,j]+L[n,i+1,j])/(hx**2)
            dL_dyy = (L[n,i,j-1]-2*L[n,i,j]+L[n,i,j+1])/(hy**2)
            # print("dNdx",dN_dxx,"dNdy",dN_dyy)
            # print("dLdx",dL_dxx,"dLdy",dL_dyy)
            print(i,j,N[n,i,j],L[n,i,j])
            f1 = alpha_N*hill_equation(N[n,i,j],L[n,i,j])
            f2 = alpha_L* hill_equation(N[n,i,j],L[n,i,j])
            # print(f1,f2)
            N[n+1,i,j] = N[n,i,j] + ht*(f1 - gamma_L*L[n,i,j] + D_N*(dN_dxx+dN_dyy))
            L[n+1,i,j] = L[n,i,j] + ht*(f2 - gamma_N*N[n,i,j] + D_L*(dL_dxx+dL_dyy))
    plt.clf()
    plt.imshow(N[n,:,:])
    plt.colorbar()
    # plt.pause(0.5)
    plt.show()