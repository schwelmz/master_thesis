import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from computation import compute_solution
import settings.settings as settings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

def convergence_test(args):
    fig, axs = plt.subplots(2,1)
    u_final_ref, v_final_ref = compute_solution(1e-5, args.outdir, args.initialization, args.videomode, args.model, "strang_EE_IE", args.dimensionless)
    axs[0].plot(np.linspace(0,100,u_final_ref.shape[0]), u_final_ref, color="black", linestyle="--")

    cmap1 = ["blue", "orange", "green", "brown"]
    linestyles = ["-", "--", "-", "--"]
    i = 0
    num_lines = 2
    for timedisc in ["EE","H","strang_EE_IE","strang_H_CN"]:
        errors = []
        for dt in [1e-2, 1e-3, 1e-4]:
            print(f"\n {i}")
            u_final, v_final = compute_solution(dt, args.outdir, args.initialization, args.videomode, args.model, timedisc, args.dimensionless)
            axs[0].plot(np.linspace(0,100,u_final.shape[0]), u_final, color=cmap1[i])
            errors.append(sp.linalg.norm(u_final - u_final_ref))
        axs[1].plot(np.arange(num_lines+1), errors, label=timedisc, color=cmap1[i])
        i += 1
    axs[1].set_yscale("log")
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def phase_diagram(args):
    max_val_N = 40
    max_val_L = 40
    # points = [(5,8), (7,8), (15.74,8), (25,8), (10,4), (10,6.45), (10,13), (10,15)]
    # points = [(4,5), (4,10), (4,40), (1,20), (10,20), (15,20)]
    # labels = ['(a)', '(b)','(c)','(d)','(e)','(f)','(g)','(h)',]
    # labels = ['(a)', '(b)','(c)','(d)','(e)','(f)']
    phase_diagram = np.load(f"out/{args.outdir}/data/phase_diagram.npy")#[0:10,0:10]
    plt.imshow(phase_diagram.T, extent=[0,max_val_N,0,max_val_L],origin="lower",cmap="gnuplot")
    # for (x,y), label in zip(points, labels):
    #     plt.text(x,y,label, color="white", ha="center", va="top", fontsize="16")
    plt.xlabel(r"$\alpha_N$")
    plt.ylabel(r"$\alpha_L$")
    # plt.axhline(y=8,color="red")
    plt.grid()
    cb = plt.colorbar()
    cb.set_label(r"$\max(N)-\min(N)$")
    # plt.gca().set_aspect(0.2)
    plt.savefig(f"out/{args.outdir}/data/phase_diagram_parameterReversed.png")
    plt.savefig(f"../../thesis/figures/phase_diagram_parameterReversed.png")
    # plt.savefig(f"out/{outdir}/data/phase_diagram_parameterReversed_zoomin.png")
    # plt.savefig(f"../../thesis/figures/phase_diagram_parameterReversed_zoomin.png")
    # plt.savefig(f"out/{outdir}/data/phase_diagram_parameter2.png")
    # plt.savefig(f"../../thesis/figures/phase_diagram_parameter2.png")
    plt.show()

def diffusion_ratio_test(args):
    plt.rcParams.update({'font.size': 20})
    val_diffs_mat = np.load(f"out/{args.outdir}/data/diffusion_rates_test.npy")
    dmin = 0
    dmax = 100
    d_vals = np.linspace(dmin,dmax,val_diffs_mat.shape[1])
    N = val_diffs_mat.shape[0]-1
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("jet")
    # for k in range(N-2):
    ax.plot(d_vals, val_diffs_mat[0,:], color="blue")    #color=cmap(k/(N-1)))
    ax.set_xlabel(r"$d=\frac{D_L}{D_N}$")
    ax.set_ylabel(r"$N_\text{max}-N_\text{min}$")
    ax.set_xticks(np.arange(0,100,20))
    ax.grid()
    # Zoomed-in inset
    # inset_ax = inset_axes(ax, width=1.5, height=1, loc="lower right", bbox_to_anchor=(0.9, 0.15,1,1))  # Adjust size and location
    # inset_ax = inset_axes(ax, width="30%", height="30%", loc="lower right",
    #                   bbox_to_anchor=(0.8, 0.1, 1, 1), bbox_transform=ax.transAxes)
    inset_ax = plt.axes([0, 0, 1, 1])  # Create an inset axes
    inset_ax.set_axes_locator(InsetPosition(ax, [0.65, 0.25, 0.3, 0.3]))
    mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="gray", lw=1)
    inset_ax.plot(d_vals, val_diffs_mat[0,:], color='blue')
    inset_ax.set_xlim(18,22)  # X range for the zoom
    inset_ax.set_xticks(np.arange(18,22,2))
    inset_ax.set_ylim(-0.1, 0.2)  # Y range for the zoom
    # inset_ax.set_title("Zoomed In")
    inset_ax.grid()

    plt.tight_layout()
    plt.savefig(f"../../thesis/figures/diffusionFactorTest_parameter2.png")
    plt.show()

if __name__ == "__main__":
    args = settings.read_cmdline_args()
    # convergence_test(args)
    phase_diagram(args)
    # diffusion_ratio_test(args)