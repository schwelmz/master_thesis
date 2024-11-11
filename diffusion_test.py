import numpy as np
import matplotlib.pyplot as plt
import settings
import time
import visualization as vis
import os
import shutil
import scipy.sparse as sparse
from functools import lru_cache

def initialize_matrix(rows, option="white-noise"):
    if option == "spike":
        array = np.zeros(rows)
        array[rows//2] = 1000
        return array
    if option == "white-noise":
        return np.random.rand(rows)

def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    return zahler/nenner

def rhs_ReceptorComplex(C_NR,C_LR,N,L,R_N,R_L):
    rhs_C_NR = k_on_N*R_N*N - k_off_N*C_NR
    rhs_C_LR = k_on_L*R_L*L - k_off_L*C_LR
    return rhs_C_NR, rhs_C_LR

def reaction_NL_modified(N, L, R_N, R_L, C_NR, C_LR):
    hill_term = hill_equation(N,L)
    reaction_N = alpha_N*hill_term - gamma_N*N - k_on_N*R_N*N + k_off_N*C_NR
    reaction_L = alpha_L*hill_term - gamma_L*L - k_on_L*R_L*L + k_off_L*C_LR
    # reaction_N = alpha_N*hill_term - gamma_N*N
    # reaction_L = alpha_L*hill_term - gamma_L*L    
    # reaction_N = - k_on_N*R_N*N + k_off_N*C_NR
    # reaction_L = - k_on_L*R_L*L + k_off_L*C_LR
    return reaction_N, reaction_L

def reaction_NL(N, L):
    hill_term = hill_equation(N,L)
    alpha_N = 4
    alpha_L = 4
    reaction_N = alpha_N*hill_term - gamma_N*N
    reaction_L = alpha_L*hill_term - gamma_L*L
    return reaction_N, reaction_L

def central_differences(U):
    # return (U[0:-2,1:-1] + U[2:,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] - 4*U[1:-1,1:-1])/(hx**2)
    return (U[0:-2] - 2*U[1:-1] + U[2:])/(hx**2)

if __name__ == '__main__':
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

    #create output directory
    if not os.path.exists(f"out/{outdir}"):
        os.makedirs(f"out/{outdir}/plots")
        os.makedirs(f"out/{outdir}/data")
    else: 
        shutil.rmtree(f"out/{outdir}")
        print(f"old output directory '{outdir}' deleted")
        os.makedirs(f"out/{outdir}/plots")
        os.makedirs(f"out/{outdir}/data")

    #model parameters
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
    d = float(parameters['d'])
    k_on_N = float(parameters['k_on_N'])
    k_on_L = float(parameters['k_on_L'])
    k_off_N = float(parameters['k_off_N'])
    k_off_L = float(parameters['k_off_L'])
    R_N_total = float(parameters['R_N_total'])
    R_L_total = float(parameters['R_L_total'])
    print("eff. diffusion Nodal = ",k_off_N/(k_on_N+k_off_N)*d)
    print("eff. diffusion Lefty = ",k_off_L/(k_on_L+k_off_L)*d)

    #discretization
    cell_length=11
    cell_number = 100
    xstart = 0
    xend = cell_length*cell_number    #100 cells of length 11µm
    tstart = 0
    tend = 50
    Nx = cell_number+1
    Nt = int(1e5)

    #Define the spatial and temporal grid
    hx = (xend-xstart)/(Nx-1)
    ht = (tend-tstart)/(Nt-1)
    xs = np.linspace(xstart,xend,Nx)
    print(f"hx={hx:.2e}, ht={ht:.2e}")
    print(f"Nx={Nx}, Nt={Nt}")

    #save 250 images in total spread evenly accross the timesteps
    frames = 250
    frameskips = Nt//frames

    #initial data
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
    #additional parameters
    C_NR_init = 0
    C_LR_init = 0

    #main loop
    tik = time.time()
    N_new = np.zeros(Nx)
    L_new = np.zeros(Nx)
    N_old = A_init
    L_old = B_init
    N_new_diff = np.zeros(Nx)
    L_new_diff = np.zeros(Nx)
    N_old_diff = A_init
    L_old_diff = B_init
    C_NR_old = C_NR_init
    C_LR_old = C_LR_init
    for n in range(0,Nt-1):
        print(f"\rtime step {n+1}/{Nt}",end=" ",flush=True)
        #update receptor binding
        R_N_old = R_N_total - C_NR_old
        R_L_old = R_L_total - C_LR_old
        rhs_C_NR, rhs_C_LR = rhs_ReceptorComplex(C_NR_old, C_LR_old, N_old, L_old, R_N_old, R_L_old)
        C_NR_new = C_NR_old + ht*rhs_C_NR
        C_LR_new = C_LR_old + ht*rhs_C_LR
        R_N_new = R_N_total - C_NR_new
        R_L_new = R_L_total - C_LR_new

        #update Nodal and Lefty
        N_total = N_old + C_NR_old
        L_total = L_old + C_LR_old
        hill_term = hill_equation(C_NR_new,C_LR_new)
        reaction_N = alpha_N*hill_term - gamma_N*N_old - k_on_N*R_N_new*N_old + k_off_N*C_NR_new
        reaction_L = alpha_L*hill_term - gamma_L*L_old - k_on_L*R_L_new*L_old + k_off_L*C_LR_new
        # reaction_N, reaction_L = reaction_NL_modified(N_old, L_old, R_N_new, R_L_new, C_NR_new, C_LR_new)
        Lap_N = central_differences(N_old)
        Lap_L = central_differences(L_old)
        N_new[1:-1] = N_old[1:-1] + ht*(reaction_N[1:-1] + d*Lap_N)
        L_new[1:-1] = L_old[1:-1] + ht*(reaction_L[1:-1] + d*Lap_L)
        #boundary conditions
        N_new[0] = N_new[1]     #left
        L_new[0] = L_new[1]
        N_new[-1] = N_new[-2]   #right
        L_new[-1] = L_new[-2]

        #update pure diffusion model
        reaction_N_diff, reaction_L_diff = reaction_NL(N_old_diff, L_old_diff)
        Lap_N_diff = central_differences(N_old_diff)
        Lap_L_diff = central_differences(L_old_diff)
        # N_new_diff[1:-1] = N_old_diff[1:-1] + ht*(D_N*Lap_N_diff)
        # L_new_diff[1:-1] = L_old_diff[1:-1] + ht*(D_L*Lap_L_diff)
        N_new_diff[1:-1] = N_old_diff[1:-1] + ht*(reaction_N_diff[1:-1] + D_N*Lap_N_diff)
        L_new_diff[1:-1] = L_old_diff[1:-1] + ht*(reaction_L_diff[1:-1] + D_L*Lap_L_diff)
        #boundary conditions
        N_new_diff[0] = N_new_diff[1]     #left
        L_new_diff[0] = L_new_diff[1]
        N_new_diff[-1] = N_new_diff[-2]   #right
        L_new_diff[-1] = L_new_diff[-2]

        #save plots
        if videomode:
            if n%frameskips == 0 or n in[1,2,3,4,5,10,40,80,150]:
                #save fig
                fig, axs = plt.subplots(4,1,figsize=(10,12))
                axs[0].plot(xs,N_new,color="red",label="Nodal")
                axs[0].plot(xs,L_new,color="blue",label="Lefty")
                axs[1].plot(xs,N_new,color="red",label="Nodal")
                axs[1].plot(xs,N_new_diff,color="red",linestyle="--",label="old model")
                axs[2].plot(xs,L_new,color="blue",label="Lefty")
                axs[2].plot(xs,L_new_diff,color="blue",linestyle="--",label="old model")
                axs[3].plot(xs,C_NR_new,color="red",label="NR-Complex")
                axs[3].plot(xs,C_LR_new,color="blue",label="LR-Complex")
                # axs[4].plot(xs, R_N_new,color="red",label="free Nodal Receptors")
                # axs[4].plot(xs, R_L_new,color="blue",label="free Nodal Receptors")
                # axs.set_ylim(0,np.maximum(np.max(N_new),np.max(L_new)))
                axs[0].legend()
                axs[1].legend()
                axs[2].legend()
                axs[3].legend()
                axs[0].set_xticks(np.arange(0,xend,100))
                axs[1].set_xticks(np.arange(0,xend,100))
                axs[2].set_xticks(np.arange(0,xend,100))
                axs[3].set_xticks(np.arange(0,xend,100))
                axs[0].set_xticks(xs, minor=True)
                axs[1].set_xticks(xs, minor=True)
                axs[2].set_xticks(xs, minor=True)
                axs[3].set_xticks(xs, minor=True)
                axs[0].grid(axis="x")
                axs[1].grid(axis="x")
                axs[2].grid(axis="x")
                axs[3].grid(axis="x")
                # axs[4].legend()
                # axs[0].set_ylim(0,np.max(L_new))
                fig.savefig(f"out/{outdir}/plots/lineplot_{n}")
                plt.close()
        #set new state as old
        N_old = N_new
        L_old = L_new
        N_old_diff = N_new_diff
        L_old_diff = L_new_diff
        C_NR_old = C_NR_new
        C_LR_old = C_LR_new
    #print computation time
    tok = time.time()
    print(f"\ndone! time taken: {(tok-tik):.1f}s")
