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

def hill_equation(A,K,n):
    # nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    # zahler = pow(N,n_N)
    # return zahler/nenner
    return pow(A,n)/(pow(K,n)+pow(A,n))

def central_differences(U):
    # return (U[0:-2,1:-1] + U[2:,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] - 4*U[1:-1,1:-1])/(hx**2)
    return (U[0:-2] - 2*U[1:-1] + U[2:])/(hx**2)

def rhs_N_f(N_f_old, params):
    N_b_old, L_b_old, R_N_old, Lap, L_f_old = params
    hill_term = hill_equation(N_b_old, K_N, n_N)
    # return hill_term + d*Lap - gamma_N*N_f_old - k_on_N*R_N_old*N_f_old + k_off_N*N_b_old
    return hill_term + d*Lap - k_on_N*R_N_old*N_f_old + k_off_N*N_b_old #- k_p*N_f_old*L_f_old
    # return d*Lap - k_on_N*R_N_old*N_f_old + k_off_N*N_b_old

def rhs_L_f(L_f_old, params):
    N_b_old, L_b_old, R_L_old, Lap, N_f_old = params
    hill_term = hill_equation(N_b_old, K_N, n_N)
    # return hill_term + d*Lap - gamma_L*L_f_old - k_on_L*R_L_old*L_f_old + k_off_L*L_b_old
    return hill_term + d*Lap - k_on_L*R_L_old*L_f_old + k_off_L*L_b_old #- k_p*N_f_old*L_f_old
    # return d*Lap - k_on_L*R_L_old*L_f_old + k_off_L*L_b_old

def rhs_N_b(N_b_old, params):
    N_f_old, R_N_old = params
    return k_on_N*R_N_old*N_f_old - k_off_N*N_b_old

def rhs_L_b(L_b_old, params):
    L_f_old, R_L_old = params
    return k_on_L*R_L_old*L_f_old - k_off_L*L_b_old

def EE(u_old, ht, rhs, params):
       return u_old + ht*rhs(u_old, params)

if __name__ == '__main__':
    # read parameters
    parameters = settings.read_parameters()
    # settings.print_settings(parameters)
    # np.random.seed(0)

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
    k_p = float(parameters['k_p'])
    n_N = float(parameters['n_N'])
    n_L = float(parameters['n_L'])
    K_N = float(parameters['K_N'])
    K_L = float(parameters['K_L'])
    gamma_N = float(parameters['gamma_N'])
    gamma_L = float(parameters['gamma_L'])
    # D_N = float(parameters['D_N'])
    # D_L = float(parameters['D_L'])
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
    tend = 100
    Nx = cell_number+1
    Nt = int(1e6)
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
        A_init = initialize_matrix(Nx, init)     #specify initial condition here
        B_init = initialize_matrix(Nx, init)     #specify initial condition here
        # A_init = init_mat 
        # B_init = init_mat
    else:
        A_init = np.load(args.input[0])
        B_init = np.load(args.input[1])
        tstart = int(args.input[2]) 
        ht = (tend-tstart)/(Nt-1)

    #main loop
    tik = time.time()
    N_f_old = A_init
    N_f_new = np.zeros(Nx)
    L_f_old = B_init
    L_f_new = np.zeros(Nx)
    N_b_old = np.zeros(Nx)
    L_b_old = np.zeros(Nx)
    R_N_old = np.ones(Nx)*R_N_total
    R_L_old = np.ones(Nx)*R_L_total
    for n in range(0,Nt-1):
        print(f"\rtime step {n+1}/{Nt}",end=" ",flush=True)

        #compute laplacian
        Lap_N_f_old = central_differences(N_f_old)
        Lap_L_f_old = central_differences(L_f_old)

        #calculate new time step
        N_f_new[1:-1] = EE(N_f_old[1:-1], ht, rhs_N_f, [N_b_old[1:-1], L_b_old[1:-1], R_N_old[1:-1], Lap_N_f_old, L_f_old[1:-1]])
        L_f_new[1:-1] = EE(L_f_old[1:-1], ht, rhs_L_f, [N_b_old[1:-1], L_b_old[1:-1], R_L_old[1:-1], Lap_L_f_old, N_f_old[1:-1]])
        N_b_new = EE(N_b_old, ht, rhs_N_b, [N_f_old, R_N_old])
        L_b_new = EE(L_b_old, ht, rhs_L_b, [L_f_old, R_L_old])
        R_N_new = R_N_total - N_b_old
        R_L_new = R_L_total - L_b_old

        #set boundary conditions
        N_f_new[0] = N_f_new[1]
        N_f_new[-1] = N_f_new[-2]
        L_f_new[0] = L_f_new[1]
        L_f_new[-1] = L_f_new[-2]

        #save plots
        if videomode:
            if n%frameskips == 0 or n in[1,2,3,4,5,10,40,80,150]:
                #save fig
                fig, axs = plt.subplots(3,1,figsize=(10,14))
                axs[0].plot(xs,N_f_new+N_b_new,color="red",label="total Nodal")
                axs[0].plot(xs,L_f_new+L_b_new,color="blue",label="total Lefty")
                axs[1].plot(xs,N_f_new,color="red",label="free Nodal")
                axs[1].plot(xs,L_f_new,color="blue",label="free Lefty")
                axs[2].plot(xs,N_b_new,color="red",label="bound Nodal")
                axs[2].plot(xs,L_b_new,color="blue",label="bound Lefty")
                # axs[1].set_ylim(0,np.max(L_f_new))
                axs[1].legend()
                axs[2].legend()
                axs[1].set_title(f"t={n}")
                fig.savefig(f"out/{outdir}/plots/lineplot_{n}")
                plt.close()

        #set new state as old
        N_f_old = N_f_new
        L_f_old = L_f_new
        N_b_old = N_b_new
        L_b_old = L_b_new
        R_N_old = R_N_new
        R_L_old = R_L_new

    #print computation time
    tok = time.time()
    print(f"\ndone! time taken: {(tok-tik):.1f}s")
