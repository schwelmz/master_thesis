import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import settings
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# read parameters
parameters = settings.read_parameters()
# settings.print_settings(parameters)

#model parameters
n_N = float(parameters['n_N'])
n_L = float(parameters['n_L'])
gamma_N = float(parameters['gamma_N'])
gamma_L = float(parameters['gamma_L'])
#dimensionless parameters:
gamma_ = gamma_L/gamma_N
print(n_N, n_L, gamma_)

f = lambda u,v: alpha_N_*u**n_N/(u**n_N + (1+v**n_L)**n_N) - u
g = lambda u,v: alpha_L_*u**n_N/(u**n_N + (1+v**n_L)**n_N) - gamma_*v

# --- Nullcline Equations (for plotting) ---
# ... (Your nullcline functions: v_func_f, u_func_g)
#nullclines calculated by hand
v_func_f = lambda u: ((u**(n_N-1)*(alpha_N_-u))**(1/n_N)-1)**(1/n_L)
u_func_g = lambda v: ( (gamma_*v*(1+v**n_L)**n_N)/(alpha_L_-gamma_*v) )**(1/n_N)

# --- Function to find intersection points (adapt from your code) ---
def find_intersections(alpha_N_, alpha_L_):
    def SoE(vars):
        u,v = vars
        return f(u,v), g(u,v)

    # Define a list to store unique steady states
    steady_states = []
    # Loop over multiple initial guesses
    guess = 100
    while guess > 1e-9:
        initial_guess = [guess, guess]
        steady_state, info, ier, msg = fsolve(SoE, initial_guess, maxfev=1000, full_output=True)
        
        # Check if the solution is unique (not already in the list)
        if ier==1 and not any(np.allclose(steady_state, s, atol=1e-3) for s in steady_states):
            steady_states.append(steady_state)
        guess = guess/10
    
    # Filter out negative solutions
    intersections = [s for s in steady_states if s[0] >= 0 and s[1] >= 0]
    return intersections

# --- Function to plot nullclines and intersections ---
def plot_nullclines(alpha_N_, alpha_L_):
    alpha_N = alpha_N_ * gamma_N
    alpha_L = alpha_L_ * gamma_N
    v_func_f_roots = [(alpha_N+np.sqrt(alpha_N**2-4))/2, (alpha_N-np.sqrt(alpha_N**2-4))/2]
    us = np.linspace(np.min(v_func_f_roots)+1e-10,np.max(v_func_f_roots),1000,dtype=np.float64)
    # us = np.linspace(np.min(v_func_f_roots),np.max(v_func_f_roots),1000,dtype=np.float64)
    vs = np.linspace(0,200,10000)
    # u_values = np.linspace(0, 5, 100)
    # v_values = np.linspace(0, 5, 100)

    intersections = find_intersections(alpha_N_, alpha_L_)

    plt.figure(figsize=(6, 6))
    plt.plot(us, v_func_f(us), label=f"Nullcline 1 (dv/dt = 0) {alpha_N_:.2f}")
    plt.plot(u_func_g(vs), vs, label=f"Nullcline 2 (du/dt = 0) {alpha_L_:.2f}")

    for u, v in intersections:
        plt.plot(u, v, 'ro', markersize=8)

    plt.xlabel("u")
    plt.ylabel("v")
    plt.title(f"Nullclines and Intersections (α_N = {alpha_N_}, α_L = {alpha_L_})")
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.legend()
    plt.grid(True)
    plt.show()

N = 200
bifurcation_diagram = np.zeros((N,N))
alpha_N_min = 0
alpha_N_max = 40
alpha_L_min = 0
alpha_L_max = 40
i = 0
for alpha_N in np.linspace(alpha_N_min,alpha_N_max,N):
  NOinter_list = []
  for alpha_L in np.linspace(alpha_L_min,alpha_L_max,N):
      alpha_N_ = alpha_N/(gamma_N)
      alpha_L_ = alpha_L/(gamma_N)
      intersections = find_intersections(alpha_N_, alpha_L_)
      NOinter = len(intersections)
      NOinter_list.append(NOinter)
  bifurcation_diagram[i,:] = NOinter_list
  i+=1

plt.rcParams.update({'font.size': 16})
# 1. Define the colors for the colormap
colors = ['lightblue', 'lightpink', 'lightgreen']
# 2. Create the colormap
cmap = ListedColormap(colors)
# Create the patches
blue_patch = mpatches.Patch(color=colors[0], label='1')
green_patch = mpatches.Patch(color=colors[1], label='2')
red_patch = mpatches.Patch(color=colors[2], label='3')

fig, axs = plt.subplots(figsize=(6,6))
axs.imshow(bifurcation_diagram.T, origin="lower", extent=[alpha_N_min,alpha_N_max,alpha_L_min,alpha_L_max], cmap=cmap)
axs.set_ylabel(r"$\alpha_L$")
axs.set_xlabel(r"$\alpha_N$")
axs.legend(handles = [blue_patch, green_patch, red_patch])
axs.grid()
fig.tight_layout()
fig.savefig("../../thesis/figures/bifurcations_numerical_2.png")
plt.show()