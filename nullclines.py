import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import settings
import sympy

plt.rcParams.update({'font.size': 16})
# read parameters
parameters = settings.read_parameters()
settings.print_settings(parameters)

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
#dimensionless parameters:
alpha_N_ = alpha_N/(gamma_N*K_N)
alpha_L_ = alpha_L/(gamma_N*K_L)
gamma_ = gamma_L/gamma_N
d = D_L/D_N
print(f"alpha_N_ = {alpha_N_}")
print(f"alpha_L_ = {alpha_L_}")
print(f"gamma_ = {gamma_}")
print(f"d = {d}")

f = lambda u,v: alpha_N_*u**n_N/(u**n_N + (1+v**n_L)**n_N) - u
g = lambda u,v: alpha_L_*u**n_N/(u**n_N + (1+v**n_L)**n_N) - gamma_*v
# print(f(0.1535,0.0001))
# u = np.linspace(0,1,100)
# v = np.linspace(0,1,100)
# plt.plot(u, f(u,0.0001))
# plt.axhline(y=0,linestyle="--",color="black")
# plt.grid()
# plt.show()

# compute steady states ##########################################################
def comp_steadyStates(f,g):
    def SoE(vars):
        u,v = vars
        return f(u,v), g(u,v)

    # Define a list to store unique steady states
    steady_states = []
    # Loop over multiple initial guesses
    search_range = np.arange(0,10,200)
    for initial_guess in [ [0,0], [1e-3, 1e-3], [1e-2, 1e-2], [0.1,0.1], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [10.0, 10.0], [100.0, 100.0], [500.0, 500.0], [1000.0,1000.0]]:
        steady_state, info, ier, msg = sp.optimize.fsolve(SoE, initial_guess, maxfev=1000, full_output=True)
        
        # Check if the solution is unique (not already in the list)
        if ier==1 and not any(np.allclose(steady_state, s, atol=1e-3) for s in steady_states):
            steady_states.append(steady_state)
    return steady_states

#stability analysis #################################################################
# Define symbols for u and v
def stability_analyis(steady_states):
    u, v = sympy.symbols('u v')

    # Define the expressions for f and g
    f_expr = alpha_N_ * u**n_N / (u**n_N + (1 + v**n_L)**n_N) - u
    g_expr = alpha_L_ * u**n_N / (u**n_N + (1 + v**n_L)**n_N) - gamma_ * v

    # Compute partial derivatives
    df_du = sympy.diff(f_expr, u)
    df_dv = sympy.diff(f_expr, v)
    dg_du = sympy.diff(g_expr, u)
    dg_dv = sympy.diff(g_expr, v)

    # Substitute steady state values
    steady_states_u = []
    steady_states_v = []
    steady_states_u_stable = []
    steady_states_u_unstable = []
    steady_states_v_stable = []
    steady_states_v_unstable = []
    for steady_state in steady_states:
        u_val, v_val = steady_state
        jacobian = np.array([[df_du.subs({u: u_val, v: v_val}), df_dv.subs({u: u_val, v: v_val})],
                            [dg_du.subs({u: u_val, v: v_val}), dg_dv.subs({u: u_val, v: v_val})]])

        # Convert to numerical values and calculate eigenvalues
        jacobian = np.array(jacobian).astype(float)
        eigenvalues = np.linalg.eigvals(jacobian)
        steady_states_u.append(steady_state[0])
        steady_states_v.append(steady_state[0])
        if eigenvalues[0] > 0 or eigenvalues[1]>0:
            stable = False
            steady_states_u_unstable.append(steady_state[0])
            steady_states_v_unstable.append(steady_state[1])
        else:
            stable = True
            steady_states_u_stable.append(steady_state[0])
            steady_states_v_stable.append(steady_state[1])
        # print("Jacobian of",steady_state,"has eigenvalues", eigenvalues, "stable:", stable)
    return steady_states_u_stable, steady_states_v_stable, steady_states_u_unstable, steady_states_v_unstable

#plot the nullclines ################################################################
#nullclines calculated by hand
v_func_f = lambda u: ((u**(n_N-1)*(alpha_N_-u))**(1/n_N)-1)**(1/n_L)
v_func_f_roots = [(alpha_N_+np.sqrt(alpha_N_**2-4))/2, (alpha_N_-np.sqrt(alpha_N_**2-4))/2]
print(v_func_f_roots)
u_func_g = lambda v: ( (gamma_*v*(1+v**n_L)**n_N)/(alpha_L_-gamma_*v) )**(1/n_N)
# Define the grid of u and v values
# u_max = np.max(steady_states_u)
# v_max = np.max(steady_states_v)
u_max = 1
v_max = 1
us = np.linspace(np.min(v_func_f_roots)+1e-10,np.max(v_func_f_roots),1000,dtype=np.float64)
vs = np.linspace(0,3,100000)
# u_values = np.linspace(0, 50000, 5000)  # Adjust the range as needed
# v_values = np.linspace(0, 20, 5000)
# U, V = np.meshgrid(u_values, v_values)
# F = f(U,V)
# G = g(U,V)
# Plot the nullclines where F = 0 (u-nullcline) and G = 0 (v-nullcline)
plt.figure()
# plt.contour(U, V, F, levels=[0], colors='blue', linewidths=2, linestyles='-')
# plt.contour(U, V, G, levels=[0], colors='orange', linewidths=2, linestyles='-')
Nlines = 5
cmap1 = plt.get_cmap('cool')(np.linspace(0,1,Nlines))
# for alpha_L in np.linspace(5,20,Nlines):
if True:
    for alpha_N in [8]:
        alpha_N_ = alpha_N/(gamma_N*K_N)
        v_func_f_roots = [(alpha_N_+np.sqrt(alpha_N_**2-4))/2, (alpha_N_-np.sqrt(alpha_N_**2-4))/2]
        us = np.linspace(np.min(v_func_f_roots)+1e-10,np.max(v_func_f_roots),1000,dtype=np.float64)
        print(f"alpha_N_ = {alpha_N_}, alpha_L = {alpha_N}")
        plt.plot(us, v_func_f(us), color="orange", linestyle="-", linewidth=2, label=fr"f(u,v)=0, $\alpha_N$={alpha_N}",zorder=1)
        k=0
        for alpha_L in [5,15.74,40]:
            alpha_L_ = alpha_L/(gamma_N*K_L)
            print(f"alpha_L_ = {alpha_L_}, alpha_L = {alpha_L}")
            #plot nullclines
            plt.plot(u_func_g(vs), vs, linestyle="-", linewidth=2, color=cmap1[k], label=fr"g(u,v)=0, $\alpha_L$={alpha_L}",zorder=1)
            #compute steady states
            ss = comp_steadyStates(f,g)
            steady_states_u_stable, steady_states_v_stable, steady_states_u_unstable, steady_states_v_unstable = stability_analyis(ss)
            plt.scatter(steady_states_u_stable, steady_states_v_stable, color="green",zorder=2)
            plt.scatter(steady_states_u_unstable, steady_states_v_unstable, color="red",zorder=2)
            print(f"Number of steady states: {len(steady_states_u_stable) + len(steady_states_u_unstable)}")
            k+=1
    #sve plot
    plt.xlabel('u')
    plt.ylabel('v')
    plt.xlim(-0.1,2.25)
    plt.ylim(-0.1,2.25)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../../thesis/figures/nullclines_parameter2_alphaLrange.png")
plt.show()

Nlines = 4
cmap2 = plt.get_cmap('autumn')(np.linspace(0,1,Nlines))
if True:
    for alpha_L in [10]:
        alpha_L_ = alpha_L/(gamma_N*K_L)
        print(f"alpha_L_ = {alpha_L_}, alpha_L = {alpha_L}")
        #plot nullcline
        plt.plot(u_func_g(vs), vs, linestyle="-", linewidth=2, color="blue", label=fr"g(u,v)=0, $\alpha_L$={alpha_L}",zorder=1)
        h=0
        for alpha_N in [4,6.45,15]:
            alpha_N_ = alpha_N/(gamma_N*K_N)
            v_func_f_roots = [(alpha_N_+np.sqrt(alpha_N_**2-4))/2, (alpha_N_-np.sqrt(alpha_N_**2-4))/2]
            us = np.linspace(np.min(v_func_f_roots)+1e-10,np.max(v_func_f_roots),1000,dtype=np.float64)
            print(f"alpha_N_ = {alpha_N_}, alpha_L = {alpha_N}")
            #plot nullcline
            plt.plot(us, v_func_f(us), color=cmap2[h], linestyle="-", linewidth=2, label=fr"f(u,v)=0, $\alpha_N$={alpha_N}",zorder=1)
            #compute steady states
            ss = comp_steadyStates(f,g)
            steady_states_u_stable, steady_states_v_stable, steady_states_u_unstable, steady_states_v_unstable = stability_analyis(ss)
            plt.scatter(steady_states_u_stable, steady_states_v_stable, color="green",zorder=2)
            plt.scatter(steady_states_u_unstable, steady_states_v_unstable, color="red",zorder=2)
            print(f"Number of steady states: {len(steady_states_u_stable) + len(steady_states_u_unstable)}")
            h+=1
    #sve plot
    plt.xlabel('u')
    plt.ylabel('v')
    plt.xlim(-0.1,2.25)
    plt.ylim(-0.1,2.25)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../../thesis/figures/nullclines_parameter2_alphaNrange.png")
plt.show()