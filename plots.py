import numpy as np
import matplotlib.pyplot as plt
import settings
import visualization as vis
import os
import shutil

plt.rcParams.update({'font.size': 20})  # Set the global font size to 12

# read parameters
parameters = settings.read_parameters()
settings.print_settings(parameters)
np.random.seed(0)

#read command line arguments
args = settings.read_cmdline_args()
setup = args.model
outdir = args.outdir

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
    # alpha_N = float(parameters['alpha_N'])
    # alpha_L = float(parameters['alpha_L'])
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
    gamma_ = gamma_L/gamma_N
    d = D_L/D_N
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

positions = ["left", "mid", "mid", "right"]
p=0
for alpha_L in[10]:
    for alpha_N in [4,6.45,13,15]:
        alpha_N_ = alpha_N/(gamma_N*K_N)
        alpha_L_ = alpha_L/(gamma_N*K_L)
        print("alpha_N_ = ",alpha_N_)
        print("alpha_L_ = ",alpha_L_)
        outdir = f"NL_parameter2_dimless_{alpha_N}_{alpha_L}"
        print(outdir)
        A = np.load(f"out/{outdir}/data/A_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")
        B = np.load(f"out/{outdir}/data/B_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")

        # if not os.path.exists(f"out/{outdir}/figures"):
        #     os.makedirs(f"out/{outdir}/figures")
        #     print("exists")
        # else: 
        #     shutil.rmtree(f"out/{outdir}/figures")
        #     print(f"old figures directory deleted")
        #     os.makedirs(f"out/{outdir}/figures")

        fig, axs = plt.subplots(constrained_layout=False)
        position = positions[p]
        if position != "right":
            colorbar = False
        else:
            colorbar = True
        img = vis.heatmap(fig,axs,A,B,Nt,[xstart,xend,ystart,yend],tstart+ht*Nt, dimless=dimless, singleplot=True, colorbar=colorbar,vmax=3)
        axs.set_title(fr"$\alpha_N$={alpha_N}, $\alpha_L$={alpha_L}")
        if position != "left":
            axs.yaxis.set_ticklabels([])
            axs.set_ylabel("")
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)
        fig.savefig(f"out/figures/phasediagram_instance_{alpha_N}_{alpha_L}.png")
        fig.savefig(f"../../thesis/figures/phasediagram_instance_{alpha_N}_{alpha_L}.png")
        p+=1