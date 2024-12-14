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
    gamma_ = gamma_L/gamma_N
    d = D_L/D_N
    print("gamma_ = ", gamma_)
    print("d = ",d)
    xstart = 0
    xend = 100
    ystart = 0
    yend = 100
    tstart = 0
    tend = 500
    Nx = 101
    Ny = 101
    Nt = int(5e4)
    dimless=True

#Define the spatial and temporal grid
hx = (xend-xstart)/(Nx-1)
hy = (xend-xstart)/(Ny-1)
ht = (tend-tstart)/(Nt-1)
xs = np.linspace(xstart,xend,Nx)
ys = np.linspace(ystart,yend,Ny)
print(f"hx={hx:.2e}, hy={hy:.2e}, ht={ht:.2e}")
print(f"Nx={Nx}, Ny={Ny}, Nt={Nt}")

#instances of phase diagram
if False:
    positions = ["left", "mid", "mid", "right"]
    p=0
    for alpha_L in[10]:
        for alpha_N in [4,6.45,13,15]:
            outdir = f"NL_parameter2_dimless_{alpha_N}_{alpha_L}"
            print(outdir)
            A = np.load(f"out/{outdir}/data/A_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")
            B = np.load(f"out/{outdir}/data/B_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")

            position = positions[p]
            if position != "right":
                fig, ax = plt.subplots(figsize=(5,5))
                img = vis.heatmap(fig,ax,A,B,Nt,[xstart,xend,ystart,yend],tstart+ht*Nt, dimless=dimless, singleplot=True,colorbar=False,vmax=3)
            else:
                fig, (ax, cax) = plt.subplots(1,2, figsize=(6,5), gridspec_kw={"width_ratios": [10,1]})
                img = vis.heatmap(fig,ax,A,B,Nt,[xstart,xend,ystart,yend],tstart+ht*Nt, dimless=dimless, singleplot=True,colorbar=False,vmax=3)
                fig.colorbar(img, cax=cax)
            ax.set_title(fr"$\alpha_N$={alpha_N}, $\alpha_L$={alpha_L}")
            plt.tight_layout()
            fig.savefig(f"out/figures/phasediagram_instance_{alpha_N}_{alpha_L}.png")
            fig.savefig(f"../../thesis/figures/phasediagram_instance_{alpha_N}_{alpha_L}.png")
            p+=1

    positions = ["left", "mid", "mid", "right"]
    p=0
    for alpha_L in[5,7,15.74,25]:
        for alpha_N in [8]:
            outdir = f"NL_parameter2_dimless_{alpha_N}_{alpha_L}"
            print(outdir)
            A = np.load(f"out/{outdir}/data/A_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")
            B = np.load(f"out/{outdir}/data/B_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")

            position = positions[p]
            if position != "right":
                fig, ax = plt.subplots(figsize=(5,5))
                img = vis.heatmap(fig,ax,A,B,Nt,[xstart,xend,ystart,yend],tstart+ht*Nt, dimless=dimless, singleplot=True,colorbar=False, vmax=3)
            else:
                fig, (ax, cax) = plt.subplots(1,2, figsize=(6,5), gridspec_kw={"width_ratios": [10,1]})
                img = vis.heatmap(fig,ax,A,B,Nt,[xstart,xend,ystart,yend],tstart+ht*Nt, dimless=dimless, singleplot=True,colorbar=False, vmax=3)
                fig.colorbar(img, cax=cax)
            ax.set_title(fr"$\alpha_N$={alpha_N}, $\alpha_L$={alpha_L}")
            plt.tight_layout()
            fig.savefig(f"out/figures/phasediagram_instance_{alpha_N}_{alpha_L}.png")
            fig.savefig(f"../../thesis/figures/phasediagram_instance_{alpha_N}_{alpha_L}.png")
        p+=1

#side by side Nodal and Lefty
if False:
    A = np.load(f"out/{outdir}/data/A_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")
    B = np.load(f"out/{outdir}/data/B_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")
    fig, axs = plt.subplots(1,2,figsize=(1.2*12,1.2*5))
    img = vis.heatmap(fig,axs,A,B,Nt,[xstart,xend,ystart,yend],tstart+ht*Nt, dimless=dimless)
    plt.tight_layout()
    # plt.show()
    fig.savefig(f"../../thesis/figures/NLsidebyside_parameter2_{alpha_N}_{alpha_L}_{tend}.png")

#instances of diffusion test
if True:
    positions = ["left", "right", "left", "right"]
    p=0
    for d in [18,20,22]:
    # for d in [100]:
        outdir = f"NL_dimless_parameter2_diffusion_{d}"
        print(outdir)
        A = np.load(f"out/{outdir}/data/A_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")
        B = np.load(f"out/{outdir}/data/B_{ht}_{hx}_{hy}_{tend}_{xend}_{yend}.npy")

        position = positions[p]
        if position != "right":
            fig, ax = plt.subplots(figsize=(5,5))
            img = vis.heatmap(fig,ax,A,B,Nt,[xstart,xend,ystart,yend],tstart+ht*Nt, dimless=dimless, singleplot=True,colorbar=False,vmax=3.5)
        else:
            fig, (ax, cax) = plt.subplots(1,2, figsize=(6,5), gridspec_kw={"width_ratios": [10,1]})
            img = vis.heatmap(fig,ax,A,B,Nt,[xstart,xend,ystart,yend],tstart+ht*Nt, dimless=dimless, singleplot=True,colorbar=False,vmax=3.5)
            fig.colorbar(img, cax=cax)
        # ax.set_title(fr"$\alpha_N$={alpha_N}, $\alpha_L$={alpha_L}")
        ax.set_title(f"d = {d}")
        plt.tight_layout()
        fig.savefig(f"out/figures/diffusion_test_instance_{d}.png")
        fig.savefig(f"../../thesis/figures/diffusion_test_instance_{d}.png")
        # fig.savefig(f"../../thesis/figures/phasediagram_instance_{alpha_N}_{alpha_L}.png")
        p+=1