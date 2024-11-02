import numpy as np
import matplotlib.pyplot as plt
import settings
import time
import visualization as vis
import os
import shutil
import scipy.sparse as sparse
from functools import lru_cache

# read parameters
parameters = settings.read_parameters()
# settings.print_settings(parameters)
np.random.seed(0)

#read command line arguments
args = settings.read_cmdline_args()
videomode = args.videomode
outdir = args.outdir
setup = args.model
time_disc = args.timedisc

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
    tstart = 0
    tend = 100/gamma_N
    Nx = 101
    Nt = int(1e4)
elif setup == "GM":     #Gierer-Meinhardt
    D_u = float(parameters["D_u"])
    D_v = float(parameters["D_v"])
    mu = float(parameters["mu"])
    a = float(parameters["a"])
    c = float(parameters["c"])
    r = float(parameters["r"])
    xstart = 0
    xend = 100
    tstart = 0
    tend = 100
    Nx = 200
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
    tstart = 0
    tend = 100
    Nx = 101
    Nt = int(1e4)
    dimless=True

#auxiliary 
@lru_cache(maxsize=8)
def make_system_matrix(N, kappa, bounds=None):
    diags = [-kappa*np.ones(N-1),(1+2*kappa)*np.ones(N), -kappa*np.ones(N-1)]
    mat = sparse.diags(diags,[-1,0,1],format="lil")
    if bounds == "neumann":
        # mat[0,1] = mat[0,0]
        # mat[-1,-2] = mat[-1,-1]
        mat[0, 0] = 1 + kappa     # Neumann boundary condition at left boundary
        mat[0, 1] = -kappa
        mat[-1, -1] = 1 + kappa   # Neumann boundary condition at right boundary
        mat[-1, -2] = -kappa
    return sparse.csr_matrix(mat)


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
    # return (U[0:-2,1:-1] + U[2:,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] - 4*U[1:-1,1:-1])/(hx**2)
    return (U[0:-2] - 2*U[1:-1] + U[2:])/(hx**2)

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
def initialize_matrix(rows, option="white-noise"):
    init_val = 1
    percentage = 0.001
    # if option == "4dots":
    #     array = np.zeros((rows,cols))
    #     d_x = cols//4
    #     d_y = rows//4
    #     array[d_x,d_y] = init_val
    #     array[3*d_x,3*d_y] = init_val
    #     array[d_x,3*d_y] = init_val
    #     array[3*d_x,d_y] = init_val
    #     return array
    # elif option == "random-dots":
    #     # Step 1: Create a 2D array filled with zeros
    #     array = np.zeros((rows, cols), dtype=int)
    #     # Step 2: Calculate the number of ones needed
    #     total_elements = rows * cols
    #     num_ones = int(total_elements * percentage)
    #     # Step 3: Randomly select positions to place the ones
    #     ones_positions = np.random.choice(total_elements, num_ones, replace=False)
    #     # Step 4: Convert 1D positions to 2D indices and place the ones
    #     for pos in ones_positions:
    #         row_index = pos // cols
    #         col_index = pos % cols
    #         array[row_index, col_index] = init_val
    #     return array
    if option == "white-noise":
        # Generate a 2D array with white noise using a normal distribution
        # low = 0.0
        # high = 0.01
        # return np.random.uniform(low, high, (rows, cols))
        return np.random.rand(rows)

'''
One time step for the Nodal-Lefty model:
Solve using 2nd order central differences in space and forward Euler in time
'''
def NodalLefty_step(N_old, L_old):
    N_new = np.zeros(Nx)
    L_new = np.zeros(Nx)
    #approximate the Laplacian operator using central differences
    Lap_N = central_differences(N_old)
    Lap_L = central_differences(L_old)
    #calculate the hill equation term
    hill_term = hill_equation(N_old[1:-1],L_old[1:-1])
    #calculate Nodal and Lefty at the new time step using the explicit euler method
    N_new[1:-1] = N_old[1:-1] + ht*(alpha_N*hill_term - gamma_N*N_old[1:-1] + D_N*Lap_N)
    L_new[1:-1] = L_old[1:-1] + ht*(alpha_L*hill_term - gamma_L*L_old[1:-1] + D_L*Lap_L)
    # set Neumann boundary values
    N_new[0] = N_new[1]     #left
    L_new[0] = L_new[1]
    N_new[-1] = N_new[-2]   #right
    L_new[-1] = L_new[-2]
    return N_new, L_new

def NodalLefty_splitting_step(N_old, L_old):
    #EE for half a time step for reaction part
    hill_term = hill_equation(N_old,L_old)
    N_half = N_old + ht/2*(alpha_N*hill_term - gamma_N*N_old)
    L_half = L_old + ht/2*(alpha_L*hill_term - gamma_L*L_old)
    #IE for one time step for diffusion part
    N_diffused = sparse.linalg.gmres(sysmat_N, N_half)
    L_diffused = sparse.linalg.gmres(sysmat_L, L_half)
    if N_diffused[1] == 0 and L_diffused[1]==0:     
        N_diffused = N_diffused[0]
        L_diffused = L_diffused[0]
    else:
        print("GMRES did not converge!")
    #EE for half a time step for reaction part
    N_new = N_diffused + ht/2*(alpha_N*hill_term - gamma_N*N_diffused)
    L_new = L_diffused + ht/2*(alpha_L*hill_term - gamma_L*L_diffused)
    return N_new, L_new

def NodalLefty_dimless_step(U_old, V_old):
    #approximate the Laplacian operator using central differences
    Lap_u = central_differences(U_old)
    Lap_v = central_differences(V_old)
    #calculate the hill equation term
    u_old = U_old[1:-1]
    v_old = V_old[1:-1]
    hill_term = pow(u_old,n_N)/(pow(u_old,n_N) + pow((1+pow(v_old,n_L)),n_N))
    #calculate Nodal and Lefty at the new time step using the explicit euler method
    u_new = u_old + ht*(alpha_N_ * hill_term - u_old + Lap_u)
    v_new = v_old + ht*(alpha_L_ * hill_term - gamma_*v_old + d*Lap_v)
    # u_new = u_old + ht*(Lap_u)
    # v_new = v_old + ht*(d*Lap_v)
    return u_new, v_new

def NodalLefty_dimless_splitting_step(N_old, L_old):
    #EE for half a time step for reaction part
    hill_term = pow(N_old,n_N)/(pow(N_old,n_N) + pow((1+pow(L_old,n_L)),n_N))
    N_half = N_old + ht/2*(alpha_N_*hill_term - N_old)
    L_half = L_old + ht/2*(alpha_L_*hill_term - gamma_*L_old)
    #IE for one time step for diffusion part
    N_diffused = sparse.linalg.gmres(sysmat_N, N_half)
    L_diffused = sparse.linalg.gmres(sysmat_L, L_half)
    if N_diffused[1] == 0 and L_diffused[1]==0:     
        N_diffused = N_diffused[0]
        L_diffused = L_diffused[0]
    else:
        print("GMRES did not converge!")
    #EE for half a time step for reaction part
    N_new = N_diffused + ht/2*(alpha_N_*hill_term - N_diffused)
    L_new = L_diffused + ht/2*(alpha_L_*hill_term - gamma_*L_diffused)
    return N_new, L_new

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

def reaction_NL(N,L):
    hill_term = hill_equation(N,L)
    reaction_N = alpha_N*hill_term - gamma_N*N
    reaction_L = alpha_L*hill_term - gamma_L*L
    return reaction_N, reaction_L

def explicit_euler(u_old, rhs, ht):
    return u_old + ht*rhs

def EE_CD_step(A, B):
    A_new = np.zeros(Nx)
    B_new = np.zeros(Nx)
    Lap_A = central_differences(A)
    Lap_B = central_differences(B)
    f_A, f_B = reaction_NL(A[1:-1],B[1:-1])
    A_new[1:-1] = explicit_euler(A[1:-1], f_A + D_N*Lap_A, ht)
    B_new[1:-1] = explicit_euler(B[1:-1], f_B + D_L*Lap_B, ht)
    # set Neumann boundary values
    A_new[0] = A_new[1]     #left
    B_new[0] = B_new[1]
    A_new[-1] = A_new[-2]   #right
    B_new[-1] = B_new[-2]
    return A_new, B_new

def implicit_euler(u, system_matrix):
    assert isinstance(system_matrix, (sparse.csr_matrix, sparse.csc_matrix)), "system matrix in wrong format!"
    u_new = sparse.linalg.spsolve(system_matrix, u)
    return u_new

def make_strang_step(int0,int1,system_matrices, reaction):
    def strang_step(u, v):
        rhsA, rhsB = reaction(u,v)
        # solve the first equation for the first half time interval
        u = int0(u, rhsA, ht/2)
        v = int0(v, rhsB, ht/2)
        # solve the second equation for one time interval
        sysmat_N, sysmat_L = system_matrices
        u = int1(u, sysmat_N)
        v = int1(v, sysmat_L)
        # solve the first equation for the second half time interval
        u = int0(u, rhsA, ht/2)
        v = int0(v, rhsB, ht/2)
        return u,v
    return strang_step

'''
Solve the PDE for a given model
'''
def stepper(integrator, A0, B0):
    #main loop
    tik = time.time()
    A_new = np.zeros(Nx)
    B_new = np.zeros(Nx)
    A_old = A0
    B_old = B0
    for n in range(0,Nt-1):
        print(f"\rtime step {n+1}/{Nt}",end=" ",flush=True)
        #update timestep
        # A_new, B_new = model_step(A_old, B_old)
        # A_new, B_new = strang_splitting(A_old, B_old, explicit_euler_step, implicit_euler_step, reaction_NL(A_new,B_new))
        A_new, B_new = integrator(A_old, B_old)
        #save plots
        if videomode:
            if n%frameskips == 0 or n in[1,2,3,4,5,10,40,80,150]:
                #save fig
                fig, axs = plt.subplots(1,1,figsize=(12,5))
                # img = vis.heatmap(fig,axs,A_new,B_new,n,[xstart,xend,ystart,yend],tstart+ht*n, dimless=dimless)
                axs.plot(xs,A_new,color="red",label="Nodal")
                axs.plot(xs,B_new,color="blue",label="Lefty")
                axs.set_ylim(0,np.maximum(np.max(A_new),np.max(B_new)))
                axs.legend()
                fig.savefig(f"out/{outdir}/plots/lineplot_{n}")
                plt.close()
                #save data
                np.save(f"out/{outdir}/data/A_{ht}_{hx}_{tend}_{xend}.npy",A_new)
                np.save(f"out/{outdir}/data/B_{ht}_{hx}_{tend}_{xend}_.npy",B_new)

        #set new state as old
        A_old = A_new
        B_old = B_new
        
    #print computation time
    tok = time.time()
    print(f"\ndone! time taken: {(tok-tik)/60:.1f}min {(tok-tik):.1f}s")
    
    return A_new, B_new

def strang_EE_IE(u0,v0):
    kappa_N = D_N*ht/(hx**2)
    kappa_L = D_L*ht/(hx**2)
    sysmat_N = make_system_matrix(Nx, kappa_N, bounds = "neumann")
    sysmat_L = make_system_matrix(Nx, kappa_L, bounds = "neumann")
    system_matrices = [sysmat_N, sysmat_L]
    return stepper(make_strang_step(explicit_euler, implicit_euler, system_matrices, reaction_NL), u0, v0)

def EE_CD(u0,v0):
    return stepper(EE_CD_step, u0, v0)

#option to continue simulation on old data, otherwise set initial conditino as "4dots", "random-dots" or "white-noise"
if args.input is None:
    if args.initialization is not None: 
        init = args.initialization
    else:
        init = "white-noise"
    init_mat = initialize_matrix(Nx, init)     #specify initial condition here
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


#Define the spatial and temporal grid
hx = (xend-xstart)/(Nx-1)
ht = (tend-tstart)/(Nt-1)
xs = np.linspace(xstart,xend,Nx)
print(f"hx={hx:.2e}, ht={ht:.2e}")
print(f"Nx={Nx}, Nt={Nt}")

#create the system matrices
if setup == "NL":
    kappa_N = D_N*ht/(hx**2)
    kappa_L = D_L*ht/(hx**2)
    sysmat_N = make_system_matrix(Nx, kappa_N, bounds="neumann")
    sysmat_L = make_system_matrix(Nx, kappa_L, bounds="neumann")
elif setup == "NL_dimless":
    kappa_N_dimless = ht/(hx**2)
    kappa_L_dimless = d*ht/(hx**2)
    sysmat_N = make_system_matrix(Nx, kappa_N_dimless, bounds="neumann")
    sysmat_L = make_system_matrix(Nx, kappa_L_dimless, bounds="neumann")

#run the simulation
if setup == "NL":
    if time_disc == "EE_CD":
        A_new, B_new = EE_CD(A_init, B_init)
    elif time_disc == "strang_EE_IE":
        A_new, B_new = strang_EE_IE(A_init, B_init)
elif setup == "GM":
    A_new, B_new = stepper(GiererMeinhardt_step)
elif setup == "NL_dimless":
    # A_new, B_new = solver(NodalLefty_dimless_step)
    A_new, B_new = stepper(NodalLefty_dimless_step)

# if np.max(A_new) > 2*np.min(A_new):
#     print("Condition for pattern formation fulfilled!")
#     val_diff = np.max(A_new) - np.min(A_new)
#     print(f"val_diff = {val_diff}")
# else:
#     print("Condition for pattern formation NOT fulfilled!")
#     val_diff = np.max(A_new) - np.min(A_new)
#     print(f"val_diff = {val_diff}")

#check for pattern formation
# N = 40
# max_val = 10
# phase_diagram = np.zeros((N,N))
# vals = np.linspace(0,max_val,N)
# # vals = [6,10]
# for i in range(N):
#     for j in range(N):
#         alpha_N = vals[i]
#         alpha_L = vals[j]
#         print(alpha_N,alpha_L)
#         if setup == "NL_dimless":
#             alpha_N_ = alpha_N/(gamma_N*K_N)
#             alpha_L_ = alpha_L/(gamma_N*K_L)
#             # print(f"alpha_N_ = {alpha_N_}, alpha_L_ = {alpha_L_}")
#         if setup == "NL":
#             A_new, B_new = solver(NodalLefty_splitting_step)
#         elif setup == "NL_dimless":
#             A_new, B_new = solver(NodalLefty_dimless_splitting_step)
#         # plt.plot(xs,A_new)
#         # plt.ylim(0,np.max(A_new))
#         # plt.show()
#         val_diff = np.max(A_new) - np.min(A_new)
#         print(f"val_diff = {val_diff}")
#         phase_diagram[i,j] = val_diff
# plt.imshow(phase_diagram, extent=[0,max_val,0,max_val],origin="lower")
# plt.xlabel(r"$\alpha_N$")
# plt.ylabel(r"$\alpha_L$")
# cb = plt.colorbar()
# cb.set_label(r"$\max(N)-\min(N)$")
# plt.show()

#save data of last time step
#save fig
fig, axs = plt.subplots(1,1,figsize=(12,5))
# img = vis.heatmap(fig,axs,A_new,B_new,n,[xstart,xend,ystart,yend],tstart+ht*n, dimless=dimless)
axs.plot(xs,A_new,color="red",label="Nodal")
axs.plot(xs,B_new,color="blue",label="Lefty")
axs.set_ylim(0,np.maximum(np.max(A_new),np.max(B_new)))
axs.legend()
fig.savefig(f"out/{outdir}/plots/lineplot_end")
np.save(f"out/{outdir}/data/A_{ht}_{hx}_{tend}_{xend}.npy",A_new)
np.save(f"out/{outdir}/data/B_{ht}_{hx}_{tend}_{xend}.npy",B_new)