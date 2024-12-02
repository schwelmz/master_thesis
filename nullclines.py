import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import settings
import sympy

# read parameters
parameters = settings.read_parameters()

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

# compute steady states ##########################################################

f = lambda u,v: alpha_N_*u**n_N/(u**n_N + (1+v**n_L)**n_N) - u
g = lambda u,v: alpha_L_*u**n_N/(u**n_N + (1+v**n_L)**n_N) - gamma_*v

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
for ss in steady_states:
    print(ss)

#stability analysis #################################################################
# Define symbols for u and v
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
    print("Jacobian of",steady_state,"has eigenvalues", eigenvalues, "stable:", stable)

#plot the nullclines ################################################################
# Define the grid of u and v values
u_max = np.max(steady_states_u)
v_max = np.max(steady_states_v)
u_values = np.linspace(0, 20, 5000)  # Adjust the range as needed
v_values = np.linspace(0, 20, 5000)
U, V = np.meshgrid(u_values, v_values)
F = f(U,V)
G = g(U,V)
# Plot the nullclines where F = 0 (u-nullcline) and G = 0 (v-nullcline)
plt.figure(figsize=(8, 6))
plt.contour(U, V, F, levels=[0], colors='blue', linewidths=1.5, linestyles='--')
plt.contour(U, V, G, levels=[0], colors='orange', linewidths=1.5, linestyles='-')
plt.scatter(steady_states_u_stable,steady_states_v_stable, marker="x", color="green")
plt.scatter(steady_states_u_unstable,steady_states_v_unstable, marker="x", color="red")
plt.xlabel('u')
plt.ylabel('v')
# plt.xlim(-0.1,u_max+10)
# plt.ylim(-0.1,v_max+10)
plt.xlim(-0.1,1)
plt.ylim(-0.1,1)
plt.title('Nullclines of the System')
plt.legend(["stable steady states", "unstable steady states"])
plt.grid()
plt.show()
