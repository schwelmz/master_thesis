import numpy as np
import matplotlib.pyplot as plt

# System parameters
D = 56.39  # Same diffusion coefficient for both
L = 100  # Domain length
dx = 0.2  # Spatial step
nx = int(L/dx) + 1  # Number of spatial points
x = np.linspace(0, L, nx)

factor = D/1.96
# Binding parameters
gammaN = 1.2/factor  # Nodal binding rate
K_N_bind = 1.0  # Nodal binding constant
gammaL = 1.0  # Lefty binding rate
K_L_bind = 1.0  # Lefty binding constant

# Production parameters
alpha_N = 10  # Base production rate for Nodal
alpha_L = 10  # Base production rate for Lefty
K_N_prod = 1.0  # Half-maximal concentration for production
K_L_prod = 1.0  # Half-maximal concentration for production
n_N = 2  # Hill coefficient for production
n_L = 2.5  # Hill coefficient for production

# Calculate stable time step
dt = 0.25 * dx * dx / D
T = 200  # Total simulation time
nt = int(T/dt)

def binding_term(C, gamma, K):
    """Receptor binding term for concentration C"""
    return -gamma * C / (C + K)

def production_term(N, L, alpha, K_N, K_L, n_N, n_L):
    """Hill-type production term activated by Nodal"""
    # return alpha * (N**n) / (K**n + N**n)
    return alpha * (N**n_N) / (N**n_N + (K_N*(1+(L/K_L)**n_L)))


# Initialize with random concentrations
np.random.seed(42)  # For reproducibility
N =0.1+0.2*np.random.rand(nx)  # Nodal: small random variations around 0.1
L =0.1+0.2*np.random.rand(nx)  # Lefty: small random variations around 0.1

# Lists to store concentration profiles over time
N_history = [N.copy()]
L_history = [L.copy()]

# Time evolution
r = D * dt / (dx * dx)  # Stability parameter
print(f"Stability parameter r = {r}")

plt.figure(figsize=(10, 6))
for step in range(nt):
    # New concentration arrays
    N_new = np.zeros(nx)
    L_new = np.zeros(nx)
    
    # Update interior points
    for i in range(1, nx-1):
        # Diffusion terms
        N_diff = r * (N[i+1] - 2*N[i] + N[i-1])
        L_diff = r * (L[i+1] - 2*L[i] + L[i-1])
        
        # Binding terms
        N_bind = dt * binding_term(N[i], gammaN, K_N_bind)
        L_bind = dt * binding_term(L[i], gammaL, K_L_bind)
        
        # Production terms
        N_prod = dt * production_term(N[i], L[i], alpha_N, K_N_prod, K_L_prod, n_N, n_L)  # Nodal autoactivation
        L_prod = dt * production_term(N[i], L[i], alpha_L, K_L_prod, K_L_prod, n_N, n_L)  # Lefty activated by Nodal
        
        # Update concentrations
        N_new[i] = N[i] + N_diff + N_bind + N_prod
        L_new[i] = L[i] + L_diff + L_bind + L_prod
    
    # No-flux boundary conditions
    N_new[0] = N_new[1]
    N_new[-1] = N_new[-2]
    L_new[0] = L_new[1]
    L_new[-1] = L_new[-2]

    #every 100th time step
    if step%100==0:
        plt.plot(x, N_new, 'b-', label='Nodal')
        plt.plot(x, L_new, 'r-', label='Lefty')
        plt.xlabel('Position')
        plt.ylabel('Concentration')
        plt.title(f'Concentration Profiles t={step*dt}/{T}')
        plt.legend()
        plt.grid(True)
        plt.pause(0.1)
        plt.clf()

    
    # Update concentrations
    N = N_new.copy()
    L = L_new.copy()
    
    # Store profiles periodically
    if step % 100 == 0:
        N_history.append(N.copy())
        L_history.append(L.copy())

# Convert to numpy arrays
N_history = np.array(N_history)
L_history = np.array(L_history)

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot concentration evolution
for i in range(len(N_history)):
    alpha = (i + 1) / len(N_history)
    ax1.plot(x, N_history[i], 'b-', alpha=alpha, linewidth=1)
    ax2.plot(x, L_history[i], 'r-', alpha=alpha, linewidth=1)

ax1.set_title('Nodal Evolution')
ax1.set_ylabel('Concentration')
ax1.grid(True)

ax2.set_title('Lefty Evolution')
ax2.set_xlabel('Position')
ax2.set_ylabel('Concentration')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Plot final state
plt.figure(figsize=(10, 6))
plt.plot(x, N_history[-1], 'b-', label='Nodal')
plt.plot(x, L_history[-1], 'r-', label='Lefty')
plt.xlabel('Position')
plt.ylabel('Concentration')
plt.title('Final Concentration Profiles')
plt.legend()
plt.grid(True)
plt.show()

# Create space-time plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

im1 = ax1.imshow(N_history, aspect='auto', origin='lower', 
                 extent=[0, L, 0, T], cmap='viridis')
plt.colorbar(im1, ax=ax1, label='Nodal Concentration')
ax1.set_xlabel('Position')
ax1.set_ylabel('Time')
ax1.set_title('Space-Time Evolution of Nodal')

im2 = ax2.imshow(L_history, aspect='auto', origin='lower', 
                 extent=[0, L, 0, T], cmap='viridis')
plt.colorbar(im2, ax=ax2, label='Lefty Concentration')
ax2.set_xlabel('Position')
ax2.set_ylabel('Time')
ax2.set_title('Space-Time Evolution of Lefty')

plt.tight_layout()
plt.show()