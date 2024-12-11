import numpy as np
import matplotlib.pyplot as plt
import settings

plt.rcParams.update({'font.size': 16})

# read parameters
parameters = settings.read_parameters()
settings.print_settings(parameters)
np.random.seed(0)

#read command line arguments
args = settings.read_cmdline_args()
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
    # d = D_L/D_N
    d = 500
    print("alpha_N_ = ",alpha_N_)
    print("alpha_L_ = ",alpha_L_)
    print("gamma_ = ", gamma_)
    # print("d = ",d)
    xstart = 0
    xend = 100
    tstart = 0
    tend = 1000     #5000
    Nx = 101
    Nt = int(1e4)
    dimless=True

max_val = 40
phase_diagram = np.load(f"out/{outdir}/data/phase_diagram.npy")#[0:10,0:10]
plt.imshow(phase_diagram, extent=[0,max_val,0,max_val],origin="lower")
plt.xlabel(r"$\alpha_L$")
plt.ylabel(r"$\alpha_N$")
# plt.axhline(y=8,color="red")
plt.grid()
cb = plt.colorbar()
cb.set_label(r"$\max(N)-\min(N)$")
plt.savefig(f"out/{outdir}/data/phase_diagram_parameter2.png")
plt.savefig(f"../../thesis/figures/phase_diagram_parameter2.png")
plt.show()