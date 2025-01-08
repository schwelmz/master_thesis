import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 24

# Parameters
L = 150  # Length of domain (μm)
dx = 0.5  # Spatial step (μm)
nx = int(L/dx) + 1  # Number of spatial points
x = np.linspace(0, L, nx)  # Spatial grid

# Create arrays for different parameter sets
params = [
    {'D': 20, 'k': 0.01, 'label': 'D = 20 μm²/s, k₁ = 0.01/s', "color": "blue", "linestyle": "-", "marker": "o"},
    {'D': 20, 'k': 0.0005, 'label': 'D = 20 μm²/s, k₁ = 0.0005/s', "color": "black", "linestyle": "-", "marker": "o"},
    {'D': 1, 'k': 0.0005, 'label': 'D = 1 μm²/s, k₁ = 0.0005/s', "color": "orange", "linestyle": "--", "marker": "x"}
]

def finite_difference(D, k, tolerance=1e-6):
    """Solve diffusion-degradation equation using finite differences until steady state."""
    # Calculate stable time step based on D
    dt = 0.25 * dx * dx / D  # Safety factor of 0.25
    
    # Initialize concentration array
    u = np.zeros(nx)
    u[0] = 1  # Boundary condition at source
    
    # Compute stability parameter
    r = D * dt / (dx * dx)
    
    print(f"For D={D}: dt={dt:.6f}, r={r:.6f}")
    
    # Time stepping with steady state check
    max_iter = 1000000  # Maximum number of iterations
    check_interval = 100  # How often to check for steady state
    
    for iter in range(max_iter):
        u_old = u.copy()
        u_new = np.zeros(nx)
        u_new[0] = 1  # Keep boundary condition fixed
        
        # Finite difference scheme
        for i in range(1, nx-1):
            u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1]) - k*dt*u[i]
        
        u_new[-1] = u_new[-2]  # Zero gradient at right boundary
        u = u_new.copy()
        
        # Check for steady state every check_interval iterations
        if iter % check_interval == 0:
            change = np.max(np.abs(u - u_old))
            if change < tolerance:
                print(f"Reached steady state after {iter} iterations for D={D}")
                break
    
    return u

def analytical_solution(x, D, k):
    """Compute analytical solution for diffusion-degradation equation."""
    lambda_val = np.sqrt(k/D)
    print(lambda_val)
    return np.exp(-lambda_val * x)

# Create figure
plt.figure(figsize=(10, 7.5))

# Plot numerical and analytical solutions for each parameter set
anal_sol_list = []
num_sol_list = []
for param in params:
    D, k = param['D'], param['k']
    print(f"D={D}, k={k}")
    
    # Compute numerical solution
    #num_sol = finite_difference(D, k)
    #num_sol_list.append(num_sol)
    
    # Compute analytical solution
    anal_sol = analytical_solution(x, D, k)
    anal_sol_list.append(anal_sol)
    
    # Plot both solutions
    #plt.plot(x, num_sol, label=f'Numerical: {param["label"]}', color=param["color"], linestyle=param["linestyle"])
    #plt.plot(x, anal_sol, label=f'Analytical: {param["label"]}', color=param["color"], linestyle=param["linestyle"])
    plt.plot(x, anal_sol, label=param["label"], color=param["color"], linestyle=param["linestyle"],linewidth=4)


#np.save("out/num_sol.npy",np.asarray(num_sol_list))
#np.save("out/anal_sol.npy",np.asarray(anal_sol_list))
plt.xlabel('Distance from source boundary (μm)')
plt.ylabel('Normalized concentration')
#plt.title('Comparison of Numerical and Analytical Solutions\nDiffusion-Degradation Equation')
plt.grid(True)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.tight_layout()
# plt.savefig("../2024_MA_Moritz_Schweller/thesis/figures/activatorInhibitorRange_B.png")
plt.show()