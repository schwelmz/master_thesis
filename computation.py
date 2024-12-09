import numpy as np
import matplotlib.pyplot as plt
import settings
import time
import visualization as vis
import os
import shutil
import scipy.sparse as sparse
from functools import lru_cache

########################################################################################
# auxiliary methods
########################################################################################
'''
Create system matrix for implicit euler solution of the diffusion equation
'''
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

########################################################################################
# reactions
########################################################################################
'''
return the values of the hill equation as it is in the PDE for different concentrations of Nodal and Lefty
'''
def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    return zahler/nenner

'''
calculate the reaction part for Nodal and Lefty respectively
'''
def reaction_NL(N,L):
    hill_term = hill_equation(N,L)
    reaction_N = alpha_N*hill_term - gamma_N*N
    reaction_L = alpha_L*hill_term - gamma_L*L
    return reaction_N, reaction_L

'''
calculate the right hand side for Nodal and Lefty respectively (reaction + diffusion)
'''
def rhs_NL(N,L):
    Lap_N = central_differences(N)
    Lap_L = central_differences(L)
    reaction_N, reaction_L = reaction_NL(N[1:-1],L[1:-1])
    rhs_N = reaction_N + D_N*Lap_N
    rhs_L = reaction_L + D_L*Lap_L
    return rhs_N, rhs_L

'''
calculate the reaction part for the dimensionless equations for Nodal and Lefty respectively
'''
def reaction_NL_dimless(N,L):
    hill_term = pow(N,n_N)/(pow(N,n_N) + pow((1+pow(L,n_L)),n_N))
    reaction_N = alpha_N_ * hill_term - N
    reaction_L = alpha_L_ * hill_term - gamma_*L
    return reaction_N, reaction_L

'''
calculate the right hand side for the dimensionless equations for Nodal and Lefty respectively (reaction + diffusion)
'''
def rhs_NL_dimless(N, L):
    Lap_N = central_differences(N)
    Lap_L = central_differences(L)
    #calculate Nodal and Lefty at the new time step using the explicit euler method
    reaction_N, reaction_L = reaction_NL_dimless(N[1:-1],L[1:-1])
    rhs_N = reaction_N + Lap_N
    rhs_L = reaction_L + d*Lap_L
    return rhs_N, rhs_L

'''
calculate the reaction parts for the Gierer Meinhardt model
'''
def reaction_GM(U,V):
    f = r*((U**2)/((1+mu*U**2)*V) - c*U)
    g = r*(U**2 - a*V)
    return f, g

'''
calculate the right hand sides for the Gierer Meinhardt model (reaction + diffusion)
'''
def rhs_GM(U,V):
    rhs_U = np.zeros(Nx)
    rhs_V = np.zeros(Nx)
    Lap_U = central_differences(U)
    Lap_V = central_differences(V)
    reaction_U, reaction_V = reaction_GM(U[1:-1],V[1:-1])
    rhs_U[1:-1] = reaction_U + D_u*Lap_U
    rhs_V[1:-1] = reaction_V + D_v*Lap_V
    return rhs_U, rhs_V

########################################################################################
# time stepping methods
########################################################################################
'''
Approximate the Laplace operator using 2nd order central differences
Input: full (Nx x Ny) matrix U (with boundary entries)
Output: approximation of the Laplace operator in each inner entry of the matrix ((Nx-2)x(Ny-2))
'''
def central_differences(U):
    # return (U[0:-2,1:-1] + U[2:,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] - 4*U[1:-1,1:-1])/(hx**2)
    return (U[0:-2] - 2*U[1:-1] + U[2:])/(hx**2)

'''
calculate one iteration of the explicit euler scheme
'''
def explicit_euler(u_old, v_old, rhs, ht):
    rhs_u, rhs_v = rhs(u_old, v_old)
    u_new = u_old + ht*rhs_u
    v_new = v_old + ht*rhs_v
    return u_new, v_new

'''
calculate one iteration of the implicit euler scheme
'''
def implicit_euler(u, v, system_matrices):
    sysmat_N, sysmat_L = system_matrices
    assert isinstance(sysmat_N, (sparse.csr_matrix, sparse.csc_matrix)), "system matrix in wrong format!"
    assert isinstance(sysmat_L, (sparse.csr_matrix, sparse.csc_matrix)), "system matrix in wrong format!"
    u_new = sparse.linalg.spsolve(sysmat_N, u)
    v_new = sparse.linalg.spsolve(sysmat_L, v)
    return u_new, v_new

'''
calculate one iteration of the heun scheme
'''
def heun(u_old, v_old, rhs, ht):
    rhs_u_old, rhs_v_old = rhs(u_old, v_old)
    u_inter = u_old + ht*rhs_u_old
    v_inter = v_old + ht*rhs_v_old
    rhs_u_inter, rhs_v_inter = rhs(u_inter, v_inter)
    u_new = u_old + ht/2*(rhs_u_old+rhs_u_inter)
    v_new = v_old + ht/2*(rhs_v_old+rhs_v_inter)
    return u_new, v_new

'''
calculate the next iteration for both species A and B and set boundaries
'''
def EE_step(A, B, rhs):
    A_new = np.zeros(Nx)
    B_new = np.zeros(Nx)
    # f_A, f_B = rhs(A,B)
    # A_new, B_new = explicit_euler(A, B, rhs, ht)
    rhs_A, rhs_B = rhs(A,B)
    A_new[1:-1] = A[1:-1] + ht*rhs_A
    B_new[1:-1] = B[1:-1] + ht*rhs_B
    # set Neumann boundary values
    A_new[0] = A_new[1]     #left
    B_new[0] = B_new[1]
    A_new[-1] = A_new[-2]   #right
    B_new[-1] = B_new[-2]
    return A_new, B_new

'''
loop over all time steps for a given integrator and right hand side
'''
def stepper(integrator, A0, B0, rhs):
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
        A_new, B_new = integrator(A_old, B_old, rhs)
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

'''
loop over all time steps for the Explicit Euler and Central Differences discretization
'''
def EE(u0,v0, rhs):
    return stepper(EE_step, u0, v0, rhs)

'''
define a strang step for 2 given integrators
'''
def make_strang_step(int0,int1,system_matrices):
    def strang_step(u, v, rhs):
        # solve the first equation for the first half time interval
        u, v = int0(u, v, rhs, ht/2)
        # solve the second equation for one time interval
        u, v = int1(u, v, system_matrices)
        # solve the first equation for the second half time interval
        u, v = int0(u, v, rhs, ht/2)
        return u,v
    return strang_step

'''
loop over all time steps for the strang splitting method using Explicit Euler and Implicit Euler methods
'''
def strang_EE_IE(u0,v0, rhs):
    return stepper(make_strang_step(explicit_euler, implicit_euler, system_matrices), u0, v0, rhs)

'''
loop over all time steps for the strang splitting method using Heun's and Implicit Euler methods
'''
def strang_H_IE(u0,v0,rhs):
    return stepper(make_strang_step(heun, implicit_euler, system_matrices), u0, v0, rhs)

########################################################################################
# main
########################################################################################
if __name__ == '__main__':
    # read parameters
    parameters = settings.read_parameters()
    settings.print_settings(parameters)
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
        tend = 500     #5000
        Nx = 101
        Nt = int(1e4)
        dimless=True

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
        system_matrices = [sysmat_N, sysmat_L]
    elif setup == "NL_dimless":
        kappa_N_dimless = ht/(hx**2)
        kappa_L_dimless = d*ht/(hx**2)
        sysmat_N = make_system_matrix(Nx, kappa_N_dimless, bounds="neumann")
        sysmat_L = make_system_matrix(Nx, kappa_L_dimless, bounds="neumann")
        system_matrices = [sysmat_N, sysmat_L]
    elif setup == "GM":
        kappa_u = D_u*ht/(hx**2)
        kappa_v = D_v*ht/(hx**2)
        sysmat_u = make_system_matrix(Nx, kappa_u, bounds="neumann")
        sysmat_v = make_system_matrix(Nx, kappa_v, bounds="neumann")
        system_matrices = [sysmat_u, sysmat_v]

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

    #run the simulation
    if False:
        if setup == "NL":
            if time_disc == "EE":
                A_new, B_new = EE(A_init, B_init, rhs_NL)
            elif time_disc == "strang_EE_IE":
                A_new, B_new = strang_EE_IE(A_init, B_init, reaction_NL)
            elif time_disc == "strang_H_IE":
                A_new, B_new = strang_H_IE(A_init, B_init, reaction_NL)
        elif setup == "GM":
            if time_disc == "EE_CD":
                A_new, B_new = EE_CD(A_init, B_init, rhs_GM)
            elif time_disc == "strang_EE_IE":
                A_new, B_new = strang_EE_IE(A_init, B_init, reaction_GM)
            elif time_disc == "strang_H_IE":
                A_new, B_new = strang_H_IE(A_init, B_init, reaction_GM)
        elif setup == "NL_dimless":
            if time_disc == "EE":
                A_new, B_new = EE(A_init, B_init, rhs_NL_dimless)
            elif time_disc == "strang_EE_IE":
                A_new, B_new = strang_EE_IE(A_init, B_init, reaction_NL_dimless)
            elif time_disc == "strang_H_IE":
                A_new, B_new = strang_H_IE(A_init, B_init, reaction_NL_dimless)
        else:
            print("Combination of model and discretization method not supported!")

        #save data of last time step
        fig, axs = plt.subplots(1,1,figsize=(12,5))
        axs.plot(xs,A_new,color="red",label="species A")
        axs.plot(xs,B_new,color="blue",label="species B")
        axs.set_ylim(0,np.maximum(np.max(A_new),np.max(B_new)))
        axs.legend()
        fig.savefig(f"out/{outdir}/plots/lineplot_end")
        np.save(f"out/{outdir}/data/A_{ht}_{hx}_{tend}_{xend}.npy",A_new)
        np.save(f"out/{outdir}/data/B_{ht}_{hx}_{tend}_{xend}.npy",B_new)

    # plot phase diagram for different values of alpha_N and alpha_L
    if True:
        N = 2
        max_val = 10
        phase_diagram = np.zeros((N,N))
        vals = np.linspace(0,max_val,N)
        # vals = [6,10]
        for i in range(N):
            for j in range(N):
                alpha_N = vals[i]
                alpha_L = vals[j]
                print(alpha_N,alpha_L)
                if setup == "NL_dimless":
                    alpha_N_ = alpha_N/(gamma_N*K_N)
                    alpha_L_ = alpha_L/(gamma_N*K_L)
                    # print(f"alpha_N_ = {alpha_N_}, alpha_L_ = {alpha_L_}")
                if setup == "NL":
                    if time_disc == "EE":
                        A_new, B_new = EE(A_init, B_init, rhs_NL)
                    elif time_disc == "strang_EE_IE":
                        A_new, B_new = strang_EE_IE(A_init, B_init, reaction_NL)
                    # A_new, B_new = solver(NodalLefty_splitting_step)
                elif setup == "NL_dimless":
                    if time_disc == "EE":
                        A_new, B_new = EE(A_init, B_init, rhs_NL_dimless)
                    elif time_disc == "strang_EE_IE":
                        A_new, B_new = strang_EE_IE(A_init, B_init, reaction_NL_dimless)
                    # A_new, B_new = solver(NodalLefty_dimless_splitting_step)
                val_diff = np.max(A_new) - np.min(A_new)
                print(f"val_diff = {val_diff}")
                phase_diagram[i,j] = val_diff
        plt.imshow(phase_diagram, extent=[0,max_val,0,max_val],origin="lower")
        plt.xlabel(r"$\alpha_N$")
        plt.ylabel(r"$\alpha_L$")
        cb = plt.colorbar()
        cb.set_label(r"$\max(N)-\min(N)$")
        plt.show()

    # check pattern formation for different diffusion rates
    if False:
        d_max = 200
        N_samples = 20
        d_vals = np.linspace(100, d_max, N_samples)
        val_diffs = []
        for d in d_vals:
            print(f"d = {d}")
            # build system matrices
            kappa_N_dimless = ht/(hx**2)
            kappa_L_dimless = d*ht/(hx**2)
            sysmat_N = make_system_matrix(Nx, kappa_N_dimless, bounds="neumann")
            sysmat_L = make_system_matrix(Nx, kappa_L_dimless, bounds="neumann")
            system_matrices = [sysmat_N, sysmat_L]
            #select solver
            if time_disc == "EE_CD":
                A_new, B_new = EE_CD(A_init, B_init, rhs_NL_dimless)
            elif time_disc == "strang_EE_IE":
                A_new, B_new = strang_EE_IE(A_init, B_init, reaction_NL_dimless)
            elif time_disc == "strang_H_IE":
                A_new, B_new = strang_H_IE(A_init, B_init, reaction_NL_dimless)
            #check pattern formation
            val_diff = np.max(A_new) - np.min(A_new)
            print(f"val_diff = {val_diff}")
            val_diffs.append(val_diff)
        #plot result
        plt.plot(d_vals,val_diffs)
        plt.xlabel(r"$d=\frac{D_L}{D_N}$")
        plt.ylabel(r"$N_\text{max}-N_\text{min}$")
        plt.show()