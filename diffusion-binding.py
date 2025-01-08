import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 24

# Parameters
L = 20  # Length of domain (μm)
dx = 0.5  # Spatial step (μm)
nx = int(L/dx) + 1  # Number of spatial points
x = np.linspace(0, L, nx)  # Spatial grid

def binding_term(N, k, K):
    """Compute receptor binding term: -kN/(N + K)"""
    return -k * N / (N + K)

def simulate(D, k, K, T_total=50):
    """Simulate diffusion with receptor binding."""
    # Calculate stable time step
    dt = 0.25 * dx * dx / D
    nt = int(T_total/dt)
    
    # Initialize concentration
    N = np.zeros(nx)
    N[0] = 1  # Source boundary condition
    
    # Compute stability parameter
    r = D * dt / (dx * dx)
    
    print(f"For D={D}: dt={dt:.6f}, r={r:.6f}")
    
    # Time stepping
    for _ in range(nt):
        N_new = np.zeros(nx)
        N_new[0] = 1  # Keep boundary condition fixed
        
        # Finite difference scheme
        for i in range(1, nx-1):
            diffusion = r * (N[i+1] - 2*N[i] + N[i-1])
            binding = dt * binding_term(N[i], k, K)
            N_new[i] = N[i] + diffusion + binding
        
        N_new[-1] = N_new[-2]  # Zero gradient at right boundary
        N = N_new.copy()
    
    return N

# Test different parameter combinations
params = [
    {'D': 20, 'k': 0.5, 'K': 0.1, 'label': r'High D (20), High $\gamma$ (0.5)', "linestyle":"-", "color":"blue"},
    {'D': 20, 'k': 0.025, 'K': 0.1, 'label': r'High D (20), Low $\gamma$ (0.025)', "linestyle":"-", "color":"black"},
    {'D': 1, 'k': 0.025, 'K': 0.1, 'label': r'Low D (1), Low $\gamma$ (0.025)', "linestyle":"--", "color":"orange"}
]

# Create figure
plt.figure(figsize=(10, 7.5))

# Plot solutions for different parameter sets
for param in params:
    sol = simulate(param["D"], param["k"], param["K"])
    plt.plot(x, sol, '-', label=param['label'], linestyle=param["linestyle"], linewidth=4, color=param["color"])

plt.xlabel('Distance from source boundary (μm)')
plt.ylabel('Normalized concentration')
plt.grid(True)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.show()
