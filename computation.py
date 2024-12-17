import numpy as np
import matplotlib.pyplot as plt
import settings
import time
import visualization as vis
import os
import shutil

# read parameters
parameters = settings.read_parameters()
settings.print_settings(parameters)
np.random.seed(0)

#read command line arguments
args = settings.read_cmdline_args()
videomode = args.videomode
outdir = args.outdir
setup = args.model

#specify setup
dimless = False
if setup == "NL":   #Nodal-Lefty
    alpha_N = float(parameters['alpha_N'])
    alpha_L = float(parameters['alpha_L'])
    # k_p = float(parameters['k_p'])
    n_N = float(parameters['n_N'])
    n_L = float(parameters['n_L'])
    K_N = float(parameters['K_N'])
    K_L = float(parameters['K_L'])
    gamma_N = float(parameters['gamma_N'])
    gamma_L = float(parameters['gamma_L'])
    D_N = float(parameters['D_N'])
    D_L = float(parameters['D_L'])
    xstart = 0
    xend = 100*np.sqrt(D_N/gamma_N)
    ystart = 0
    yend = 100*np.sqrt(D_N/gamma_N)
    tstart = 0
    tend = 100/gamma_N
    Nx = 101
    Ny = 101
    Nt = int(1e5)
elif setup == "GM":     #Gierer-Meinhardt
    D_u = float(parameters["D_u"])
    D_v = float(parameters["D_v"])
    mu = float(parameters["mu"])
    a = float(parameters["a"])
    c = float(parameters["c"])
    r = float(parameters["r"])
    xstart = 0
    xend = 100
    ystart = 0
    yend = 100
    tstart = 0
    tend = 100
    Nx = 200
    Ny = 200
    Nt = int(1e4)
elif setup == "NL_dimless":     #dimensionaless Nodal-Lefty
    alpha_N = float(parameters['alpha_N'])
    alpha_L = float(parameters['alpha_L'])
    # k_p = float(parameters['k_p'])
    n_N = float(parameters['n_N'])
    n_L = float(parameters['n_L'])
    K_N = float(parameters['K_N'])
    K_L = float(parameters['K_L'])
    gamma_N = float(parameters['gamma_N'])
    gamma_L = float(parameters['gamma_L'])
    D_N = float(parameters['D_N'])
    D_L = float(parameters['D_L'])
    #dimensionless parameters:
    alpha_N_ = alpha_N/(gamma_N*K_N)
    alpha_L_ = alpha_L/(gamma_N*K_L)
    gamma_ = gamma_L/gamma_N
    d = D_L/D_N
    print("alpha_N_ = ",alpha_N_)
    print("alpha_L_ = ",alpha_L_)
    print("gamma_ = ", gamma_)
    print("d = ",d)
    xstart = 0
    xend = 100
    ystart = 0
    yend = 100
    tstart = 0
    tend = 100
    Nx = 101
    Ny = 101
    Nt = int(1e5)
    dimless=True

#Define the spatial and temporal grid
hx = (xend-xstart)/(Nx-1)
hy = (xend-xstart)/(Ny-1)
ht = (tend-tstart)/(Nt-1)
xs = np.linspace(xstart,xend,Nx)
ys = np.linspace(ystart,yend,Ny)
print(f"hx={hx:.2e}, hy={hy:.2e}, ht={ht:.2e}")
print(f"Nx={Nx}, Ny={Ny}, Nt={Nt}")

'''
return the values of the hill equation as it is in the PDE for different concentrations of Nodal and Lefty
'''
def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    return zahler/nenner

'''
Approximate the Laplace operator using 2nd order central differences
Input: full (Nx x Ny) matrix U (with boundary entries)
Output: approximation of the Laplace operator in each inner entry of the matrix ((Nx-2)x(Ny-2))
'''
def central_differences(U):
    return (U[0:-2,1:-1] + U[2:,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] - 4*U[1:-1,1:-1])/(hx**2)

# def laplacian_2d(arr, dx, dy):
#     laplacian_x = (np.roll(arr, 1, axis=0) - 2 * arr + np.roll(arr, -1, axis=0)) / dx**2
#     laplacian_y = (np.roll(arr, 1, axis=1) - 2 * arr + np.roll(arr, -1, axis=1)) / dy**2
#     return laplacian_x + laplacian_y
 
'''
Create matrix containing the initial values
Input: desired number of rows, desired number of columns
options for initialization: "4dots" (1 dot in each quarter of the matrix), 
                            "random-dots" (randomly distributed dots in <percentage> of the total values)
                            "white-noise" (random values sampled from a uniform distribution over [0,1) )
Output: matrix containing initial values
'''
def initialize_matrix(rows, cols, option="white-noise"):
    init_val = 1
    percentage = 0.001
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
        # high = 0.01
        # return np.random.uniform(low, high, (rows, cols))
        return np.random.rand(rows,cols)

'''
One time step for the Nodal-Lefty model:
Solve using 2nd order central differences in space and forward Euler in time
'''
def NodalLefty_step(N_old, L_old):
    #approximate the Laplacian operator using central differences
    Lap_N = central_differences(N_old)
    Lap_L = central_differences(L_old)
    # Lap_N = laplacian_2d(N_old,hx,hy)
    # Lap_L = laplacian_2d(L_old,hx,hy)
    #calculate the hill equation term
    hill_term = hill_equation(N_old[1:-1,1:-1],L_old[1:-1,1:-1])
    # hill_term = hill_equation(N_old,L_old)
    #calculate Nodal and Lefty at the new time step using the explicit euler method
    N_new = N_old[1:-1,1:-1] + ht*(alpha_N*hill_term - gamma_N*N_old[1:-1,1:-1] + D_N*Lap_N)
    L_new = L_old[1:-1,1:-1] + ht*(alpha_L*hill_term - gamma_L*L_old[1:-1,1:-1] + D_L*Lap_L)
    # N_new = N_old + ht*(alpha_N*hill_term - gamma_N*N_old + D_N*Lap_N)
    # L_new = L_old + ht*(alpha_L*hill_term - gamma_L*L_old + D_L*Lap_L)
    return N_new, L_new

def NodalLefty_dimless_step(U_old, V_old):
    #approximate the Laplacian operator using central differences
    Lap_u = central_differences(U_old)
    Lap_v = central_differences(V_old)
    #calculate the hill equation term
    u_old = U_old[1:-1,1:-1]
    v_old = V_old[1:-1,1:-1]
    hill_term = pow(u_old,n_N)/(pow(u_old,n_N) + pow((1+pow(v_old,n_L)),n_N))
    #calculate Nodal and Lefty at the new time step using the explicit euler method
    u_new = u_old + ht*(alpha_N_ * hill_term - u_old + Lap_u)
    v_new = v_old + ht*(alpha_L_ * hill_term - gamma_*v_old + d*Lap_v)
    # u_new = u_old + ht*(Lap_u)
    # v_new = v_old + ht*(d*Lap_v)
    return u_new, v_new

'''
One time step for the Gierer-Meinhardt model:
Solve using 2nd order central differences in space and forward Euler in time
'''
def GiererMeinhardt_step(U_old, V_old):
    #approximate the Laplacian operator using central differences
    Lap_U = central_differences(U_old)
    Lap_V = central_differences(V_old)
    u = U_old[1:-1,1:-1]
    v = V_old[1:-1,1:-1]
    f = (u**2)/((1+mu*u**2)*v) - c*u
    g = u**2 - a*v
    U_new = u + ht*(r*f + D_u*Lap_U)
    V_new = v + ht*(r*g + D_v*Lap_V)
    return U_new, V_new

'''
Solve the PDE for a given model
'''
def solver(model_step):
    #main loop
    tik = time.time()
    A_new = np.zeros((Nx,Ny))
    B_new = np.zeros((Nx,Ny))
    A_old = A_init
    B_old = B_init
    for n in range(0,Nt-1):
        print(f"\rtime step {n+1}/{Nt}",end=" ",flush=True)
        #update timestep
        A_new[1:-1,1:-1], B_new[1:-1,1:-1] = model_step(A_old, B_old)
        # A_new, B_new = model_step(A_old, B_old)
        #set Neumann boundary values
        A_new[0,:] = A_new[1,:]     #left
        B_new[0,:] = B_new[1,:]
        A_new[:,0] = A_new[:,1]     #bottom
        B_new[:,0] = B_new[:,1]
        A_new[-1,:] = A_new[-2,:]   #right
        B_new[-1,:] = B_new[-2,:]
        A_new[:,-1] = A_new[:,-2]   #top
        B_new[:,-1] = B_new[:,-2]
        #save plots
        if videomode:
            if n%frameskips == 0 or n in[1,2,3,4,5,10,40,80,150]:
                #save fig
                fig, axs = plt.subplots(1,2,figsize=(12,5))
                img = vis.heatmap(fig,axs,A_new,B_new,n,[xstart,xend,ystart,yend],tstart+ht*n, dimless=dimless)
                fig.savefig(f"out/{outdir}/plots/heatmap_{n}")
                plt.close()
                #save data
                np.save(f"out/{outdir}/data/A_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy",A_new)
                np.save(f"out/{outdir}/data/B_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy",B_new)

        #set new state as old
        A_old = A_new
        B_old = B_new
        
    #print computation time
    tok = time.time()
    print(f"\ndone! time taken: {(tok-tik)/60:.1f}min")
    
    return A_new, B_new

#option to continue simulation on old data, otherwise set initial conditino as "4dots", "random-dots" or "white-noise"
if args.input is None:
    if args.initialization is not None: 
        init = args.initialization
    else:
        init = "white-noise"
    init_mat = initialize_matrix(Nx,Ny, init)     #specify initial condition here
    A_init = init_mat 
    B_init = init_mat
else:
    A_init = np.load(args.input[0])
    B_init = np.load(args.input[1])
    tstart = int(args.input[2]) 
    ht = (tend-tstart)/(Nt-1)

#save 250 images in total spread evenly accross the timesteps
frames = 250
frameskips = Nt//frames

#create output directory
if not os.path.exists(f"out/{outdir}"):
    os.makedirs(f"out/{outdir}/plots")
    os.makedirs(f"out/{outdir}/data")
else: 
    shutil.rmtree(f"out/{outdir}")
    print(f"old output directory '{outdir}' deleted")
    os.makedirs(f"out/{outdir}/plots")
    os.makedirs(f"out/{outdir}/data")

#run the simulation
if setup == "NL":
    A_new, B_new = solver(NodalLefty_step)
elif setup == "GM":
    A_new, B_new = solver(GiererMeinhardt_step)
elif setup == "NL_dimless":
    A_new, B_new = solver(NodalLefty_dimless_step)

#save data of last time step
np.save(f"out/{outdir}/data/A_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy",A_new)
np.save(f"out/{outdir}/data/B_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy",B_new)
