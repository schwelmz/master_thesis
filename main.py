import numpy as np
import matplotlib.pyplot as plt
import settings
import scipy as sp

# read parameters
alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L = settings.read_parameters()
settings.print_settings()

xstart = 0
xend = 1
tstart = 0
tend = 0.1
hx = 1.3e-2
ht = 5e-7
Nx = int((xend-xstart)/(hx)+1)
Nt = int((tend-tstart)/(ht)+1)
xs = np.linspace(xstart,xend,Nx)
alpha = D_N*ht/hx**2

def make_laplace(N, hx = 1, bounds=None):
    if hasattr(hx, "__len__"):
        """
        0    1    2
        |----|----|----|--
        h0   h1   h2
        """
        assert(len(hx) == N - 1), f"len(hx) = {len(hx)}, N = {N}"
        h = lambda i: hx[i]
    else:
        h = lambda i: hx


    rows = []
    cols = []
    vals = []
    for i in range(N):
        if bounds=='skip' and i in [0, N-1]:
            continue
        if bounds=='dirichlet' and i in [0, N-1]:
            rows.append(i)
            cols.append(i)
            vals.append(1)
            continue
        if bounds=='neumann' and i in [0, N-1]:
            if i == 0:
                rows.append(i)
                cols.append(i)
                vals.append(-1/h(0))
                rows.append(i)
                cols.append(i+1)
                vals.append(1/h(0))
            else:
                rows.append(i)
                cols.append(i-1)
                vals.append(1/h(N-2))
                rows.append(i)
                cols.append(i)
                vals.append(-1/h(N-2))
            continue

        if i != 0:
            rows.append(i)
            cols.append(i-1)
            vals.append(1/h(i-1)) # ∇φ_i ∇φ_i-1
        rows.append(i)
        cols.append(i)
        vals.append(-1/h(i-1) - 1/h(i)) # ∇φ_i ∇φ_i
        if i != N-1:
            rows.append(i)
            cols.append(i+1)
            vals.append(1/h(i)) # ∇φ_i ∇φ_i+1
    # return as Δ =  <∇φ, ∇φ>
    return -sp.sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def hill_equation(N,L):
    nenner = pow(N,n_N) + pow(K_N * (1 + pow(L/K_L,n_L)), n_N)
    zahler = pow(N,n_N)
    return zahler/nenner

'''
Discretize equation using 
- Explicit Euler in time and 
- 2nd order Central Differences in space
'''
def EE_CD(N,L):
    N_new = np.zeros(N.shape[0])
    L_new = np.zeros(L.shape[0])
    for i in range(1,Nx-1):
        dN_dxx = (N[i-1]-2*N[i]+N[i+1])/(hx**2)
        dL_dxx = (L[i-1]-2*L[i]+L[i+1])/(hx**2)
        hill_term = hill_equation(N[i],L[i])
        N_new[i] = N[i] + ht*(alpha_N*hill_term - gamma_N*N[i] + D_N*dN_dxx)
        L_new[i] = L[i] + ht*(alpha_L*hill_term - gamma_L*L[i] + D_L*dL_dxx)
    return N_new, L_new

def explicit_euler_step(u,rhs,ht):
    return u + ht*rhs

def implicit_euler_step(u0):
    u = sp.sparse.linalg.cg(laplace, u0)
    return u[0]

def strang_EE_IE(N0,L0):
    # 1/2 time step for reaction term with EE
    N1=np.zeros(Nx)
    L1=np.zeros(Nx)
    for i in range(1,Nx-1):
        hill_term = hill_equation(N0[i],L0[i])
        rhs_N = alpha_N * hill_term - gamma_N*N0[i]
        rhs_L = alpha_L * hill_term - gamma_L*L0[i]
        N1[i] = explicit_euler_step(N0[i], rhs_N, ht/2)
        L1[i] = explicit_euler_step(L0[i], rhs_L, ht/2)
    N1[0] = N1[1]
    N1[-1] = N1[-2]

    #1 time step for diffusion with IE
    N1 = implicit_euler_step(N1)
    L1 = implicit_euler_step(L1)

    # 1/2 time step for reaction term with EE
    for i in range(1,Nx-1):
        hill_term = hill_equation(N1[i],L1[i])
        rhs_N1 = alpha_N * hill_term - gamma_N*N1[i]
        rhs_L1 = alpha_L * hill_term - gamma_N*L1[i]
        N1[i] = explicit_euler_step(N1[i], rhs_N1, ht/2)
        L1[i] = explicit_euler_step(L1[i], rhs_L1, ht/2)
    N1[0] = N1[1]
    N1[-1] = N1[-2]

    return N1, L1

def IE_CD():
    return 0

#initial condition
N = np.zeros((Nt,Nx))
L = np.zeros((Nt,Nx))
m = Nx//2
N[0,m] = 10
L[0,m] = 10

#create system matrix
laplace = make_laplace(Nx,1,bounds="dirichlet")
#print(laplace.todense())

fig, axs = plt.subplots(1,2)
# main loop
for n in range(0,Nt-1):
    #plot
    plt.clf()
    plt.plot(xs, N[n,:], label="Nodal")
    plt.plot(xs, L[n,:], label="Lefty")
    plt.legend()
    plt.ylim(0,15)
    plt.show()
    # plt.pause(0.1)

    print(n+1,"/",Nt)
    #compute solution at new time step
    N[n+1,:], L[n+1,:]= strang_EE_IE(N[n,:],L[n,:])