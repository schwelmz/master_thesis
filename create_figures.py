import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from computation import compute_solution
import settings.settings as settings

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

if __name__ == "__main__":
    args = settings.read_cmdline_args()
    compute_solution(1e-3, args.outdir, args.initialization, args.videomode, args.model, args.timedisc, args.dimensionless, int(args.dimensions))