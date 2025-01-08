import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy as sp
import scipy.sparse as sparse
from functools import lru_cache
import time
import os
import shutil
import settings

class ModelParameters:
    """Container for model parameters to avoid global variables"""
    def __init__(self, model, params_dict, is_dimensionless):
        if model == "NL":
            self._init_NL(params_dict)
            if is_dimensionless:
                self._compute_NL_dimensionless()
        elif model == "GM":
            print("ja")
            self._init_GM(params_dict)
            
    def _init_NL(self, params):
        # Nodal-Lefty parameters
        self.alpha_N = float(params.get('alpha_N', 0))
        self.alpha_L = float(params.get('alpha_L', 0))
        self.n_N = float(params.get('n_N', 0))
        self.n_L = float(params.get('n_L', 0))
        self.K_N = float(params.get('K_N', 0))
        self.K_L = float(params.get('K_L', 0))
        self.gamma_N = float(params.get('gamma_N', 0))
        self.gamma_L = float(params.get('gamma_L', 0))
        self.D_N = float(params.get('D_N', 0))
        self.D_L = float(params.get('D_L', 0))


        
    def _compute_NL_dimensionless(self):
        # Compute dimensionless parameters
        self.alpha_N_dimless = self.alpha_N / (self.gamma_N * self.K_N)
        self.alpha_L_dimless = self.alpha_L / (self.gamma_N * self.K_L)
        self.gamma_dimless = self.gamma_L / self.gamma_N
        self.d = self.D_L / self.D_N
    
    def _init_GM(self, params):
        # Gierer-Meinhardt parameters
        self.D_u = float(params.get('D_u', 0))
        self.D_v = float(params.get('D_v', 0))
        self.mu = float(params.get('mu', 0))
        self.a = float(params.get('a', 0))
        self.c = float(params.get('c', 0))
        self.r = float(params.get('r', 0))
    
    def print(self):
        print('Model parameters:')
        print('\n'.join("\t%s: %s" % (key,value) for key,value in vars(self).items()))

class GridParameters:
    """Container for spatial and temporal grid parameters"""
    def __init__(self, is_dimensionless, model_params, grid_params_dict):
        if is_dimensionless:
            self._init_dimensionless(grid_params_dict)
        else:
            self._init_dimensional(model_params, grid_params_dict)
    
    def _init_dimensionless(self, grid_params_dict):
        self.x_start = grid_params_dict["x_start"]
        self.x_end = grid_params_dict["x_end"]
        self.t_start = grid_params_dict["t_start"]
        self.t_end = grid_params_dict["t_end"]
        self.dx = grid_params_dict["dx"]
        self.dt = grid_params_dict["dt"]
        self.nx = int((self.x_end - self.x_start) / self.dx)
        self.nt = int((self.t_end - self.t_start) / self.dt)
        self.x = np.linspace(self.x_start, self.x_end, self.nx)
    
    def _init_dimensional(self, model_params, grid_params_dict):
        self.x_start = grid_params_dict["x_start"]
        self.x_end = grid_params_dict["x_end"] * np.sqrt(model_params.D_N/model_params.gamma_N)
        self.t_start = grid_params_dict["t_start"]
        self.t_end = grid_params_dict["t_end"] / model_params.gamma_N
        self.dx = grid_params_dict["dx"]
        self.dt = grid_params_dict["dt"]
        self.nx = int((self.x_end - self.x_start) / self.dx)
        self.nt = int((self.t_end - self.t_start) / self.dt)
        self.x = np.linspace(self.x_start, self.x_end, self.nx)

    def print(self):
        print('Grid parameters:')
        print('\n'.join("\t%s: %s" % (key,value) for key,value in vars(self).items() if key != "x"))

class ReactionDiffusionModel:
    """Base class for reaction-diffusion models"""
    def __init__(self, params, grid):
        self.params = params
        self.grid = grid
        self._setup_system_matrices()
        
    @property
    def labels(self):
        """Return dictionary of labels for plotting"""
        return {
            'x_label': 'x',
            'y_label': 'Concentration',
            'title': 'Reaction-Diffusion System',
            'u_label': 'Species U',
            'v_label': 'Species V'
        }

    @lru_cache(maxsize=8)
    def _create_system_matrix(self, N, kappa, bounds="neumann"):
        """Create system matrix for implicit euler solution"""
        diags = [kappa * np.ones(N-1), (-2*kappa) * np.ones(N), kappa * np.ones(N-1)]
        mat = sparse.diags(diags, [-1, 0, 1], format="lil")
        
        # if bounds == "neumann":
        #     # Left boundary: u[-1] = u[1] -> -u[-1] + 2u[0] - u[1] = 0
        #     mat[0,0] = -kappa  # Diagonal term
        #     mat[0,1] = kappa   # Off-diagonal term
            
        #     # Right boundary: u[N+1] = u[N-1] -> -u[N-1] + 2u[N] - u[N+1] = 0
        #     mat[-1,-1] = -kappa  # Diagonal term
        #     mat[-1,-2] = kappa   # Off-diagonal term
            
        return sparse.csr_matrix(mat)
    
    def _apply_bc_sysmat(self, sysmat):
        sysmat[0][0,0] = -sysmat[0][0,1]
        sysmat[0][-1,-1] = -sysmat[0][-1,-2]
        sysmat[1][0,0] = -sysmat[1][0,1]
        sysmat[1][-1,-1] = -sysmat[1][-1,-2]
        return sysmat

    def _setup_system_matrices(self):
        """To be implemented by child classes"""
        raise NotImplementedError
        
    def reaction_terms(self, u, v):
        """To be implemented by child classes"""
        raise NotImplementedError
        
    def rhs(self, u, v):
        """Calculate right-hand side including both reaction and diffusion terms"""
        raise NotImplementedError

class NodalLeftyModel(ReactionDiffusionModel):
    """Implementation of the dimensional Nodal-Lefty model"""
    @property
    def labels(self):
        """Return dictionary of labels for plotting"""
        return {
            'x_label': r'Position ($\mu m$)',
            'y_label': r'Concentration ($nM$)',
            'title': 'Nodal-Lefty System',
            'u_label': 'Nodal',
            'v_label': 'Lefty'
        }

    def _setup_system_matrices(self):
        kappa_N = self.params.D_N * self.grid.dt / (self.grid.dx ** 2)
        kappa_L = self.params.D_L * self.grid.dt / (self.grid.dx ** 2)
        I = sparse.identity(self.grid.nx, format="csr")
        self.system_matrices_ie = [
            I - self._create_system_matrix(self.grid.nx, kappa_N, "neumann"),
            I - self._create_system_matrix(self.grid.nx, kappa_L, "neumann")
        ]
        self.system_matrices_cn_implicit = [
            I - 0.5 * self._create_system_matrix(self.grid.nx, kappa_N, "neumann"),
            I - 0.5 * self._create_system_matrix(self.grid.nx, kappa_L, "neumann")
        ]
        self.system_matrices_cn_explicit = [
            I + 0.5 * self._create_system_matrix(self.grid.nx, kappa_N, "neumann"),
            I + 0.5 * self._create_system_matrix(self.grid.nx, kappa_L, "neumann")
        ]
        

    def hill_equation(self, N, L):
        """Calculate Hill equation term"""
        numerator = np.power(N, self.params.n_N)
        denominator = (np.power(N, self.params.n_N) + 
                      np.power(self.params.K_N * 
                              (1 + np.power(L/self.params.K_L, self.params.n_L)),
                              self.params.n_N))
        return numerator / denominator
        
    def reaction_terms(self, N, L):
        """Calculate reaction terms for Nodal and Lefty"""
        hill_term = self.hill_equation(N, L)
        r_N = self.params.alpha_N * hill_term - self.params.gamma_N * N
        r_L = self.params.alpha_L * hill_term - self.params.gamma_L * L
        return r_N, r_L
        
    def rhs(self, N, L):
        """Calculate complete right-hand side including diffusion"""
        diff_N = self._laplacian(N)
        diff_L = self._laplacian(L)
        r_N, r_L = self.reaction_terms(N[1:-1], L[1:-1])
        
        rhs_N = np.zeros_like(N)
        rhs_L = np.zeros_like(L)
        rhs_N[1:-1] = r_N + self.params.D_N * diff_N
        rhs_L[1:-1] = r_L + self.params.D_L * diff_L
        return rhs_N, rhs_L
    
    def _laplacian(self, U):
        """Calculate Laplacian using central differences"""
        return (U[0:-2] - 2*U[1:-1] + U[2:]) / (self.grid.dx ** 2)

class DimensionlessNodalLeftyModel(ReactionDiffusionModel):
    """Implementation of the dimensionless Nodal-Lefty model"""
    @property
    def labels(self):
        """Return dictionary of labels for plotting"""
        return {
            'x_label': r'Dimensionless Position $x^*$',
            'y_label': 'Dimensionless Concentration',
            'title': 'Dimensionless Nodal-Lefty System',
            'u_label': 'Nodal',
            'v_label': 'Lefty'
        }


    def _setup_system_matrices(self):
        kappa_N = self.grid.dt / (self.grid.dx ** 2)
        kappa_L = self.params.d * self.grid.dt / (self.grid.dx ** 2)
        I = sparse.identity(self.grid.nx, format="csr")
        self.system_matrices_ie = [
            I - self._create_system_matrix(self.grid.nx, kappa_N, "neumann"),
            I - self._create_system_matrix(self.grid.nx, kappa_L, "neumann")
        ]
        self.system_matrices_ie = self._apply_bc_sysmat(self.system_matrices_ie)

        self.system_matrices_cn_implicit = [
            I - 0.5 * self._create_system_matrix(self.grid.nx, kappa_N, "neumann"),
            I - 0.5 * self._create_system_matrix(self.grid.nx, kappa_L, "neumann")
        ]
        self.system_matrices_cn_implicit = self._apply_bc_sysmat(self.system_matrices_cn_implicit)
        self.system_matrices_cn_explicit = [
            I + 0.5 * self._create_system_matrix(self.grid.nx, kappa_N, "neumann"),
            I + 0.5 * self._create_system_matrix(self.grid.nx, kappa_L, "neumann")
        ]
        self.system_matrices_cn_explicit = self._apply_bc_sysmat(self.system_matrices_cn_explicit)

        
    def hill_equation(self, N, L):
        """Calculate dimensionless Hill equation term"""
        numerator = np.power(N, self.params.n_N)
        denominator = np.power(N, self.params.n_N) + np.power(1 + np.power(L, self.params.n_L), self.params.n_N)
        return numerator / denominator
        
    def reaction_terms(self, N, L):
        """Calculate dimensionless reaction terms"""
        hill_term = self.hill_equation(N,L)
        r_N = self.params.alpha_N_dimless * hill_term - N
        r_L = self.params.alpha_L_dimless * hill_term - self.params.gamma_dimless * L
        return r_N, r_L
        
    def rhs(self, N, L):
        """Calculate complete dimensionless right-hand side"""
        diff_N = self._laplacian(N)
        diff_L = self._laplacian(L)
        r_N, r_L = self.reaction_terms(N[1:-1], L[1:-1])
        
        rhs_N = np.zeros_like(N)
        rhs_L = np.zeros_like(L)
        rhs_N[1:-1] = r_N + diff_N
        rhs_L[1:-1] = r_L + self.params.d * diff_L
        return rhs_N, rhs_L

    def _laplacian(self, U):
        """Calculate Laplacian using central differences"""
        return (U[0:-2] - 2*U[1:-1] + U[2:]) / (self.grid.dx ** 2)

class GiererMeinhardtModel(ReactionDiffusionModel):
    """Implementation of the Gierer-Meinhardt model"""
    @property
    def labels(self):
        """Return dictionary of labels for plotting"""
        return {
            'x_label': 'Position x',
            'y_label': 'Concentration',
            'title': 'Gierer-Meinhardt System',
            'u_label': 'Species A',
            'v_label': 'Species B'
        }

    def _setup_system_matrices(self):
        kappa_u = self.params.D_u * self.grid.dt / (self.grid.dx ** 2)
        kappa_v = self.params.D_v * self.grid.dt / (self.grid.dx ** 2)
        I = sparse.identity(self.grid.nx, format="csr")
        self.system_matrices_ie = [
            I - self._create_system_matrix(self.grid.nx, kappa_u, "neumann"),
            I- self._create_system_matrix(self.grid.nx, kappa_v, "neumann")
        ]
        self.system_matrices_cn_implicit = [
            I - self._create_system_matrix(self.grid.nx, kappa_u, "neumann"),
            I- self._create_system_matrix(self.grid.nx, kappa_v, "neumann")
        ]
        self.system_matrices_cn_explicit = [
            I + self._create_system_matrix(self.grid.nx, kappa_u, "neumann"),
            I + self._create_system_matrix(self.grid.nx, kappa_v, "neumann")
        ]
        
    def reaction_terms(self, U, V):
        """Calculate reaction terms for both species"""
        r_U = self.params.r*((U**2)/((1+self.params.mu*U**2)*V) - self.params.c*U)
        r_V = self.params.r*(U**2 - self.params.a*V)
        return r_U, r_V
        
    def rhs(self, U, V):
        """Calculate complete right-hand side including diffusion"""
        diff_U = self._laplacian(U)
        diff_V = self._laplacian(V)
        r_U, r_V = self.reaction_terms(U[1:-1], V[1:-1])
        
        rhs_U = np.zeros_like(U)
        rhs_V = np.zeros_like(V)
        rhs_U[1:-1] = r_U + self.params.D_u * diff_U
        rhs_V[1:-1] = r_V + self.params.D_v * diff_V
        return rhs_U, rhs_V
    
    def _laplacian(self, U):
        """Calculate Laplacian using central differences"""
        return (U[0:-2] - 2*U[1:-1] + U[2:]) / (self.grid.dx ** 2)

class TimeStepper:
    """Handles time stepping for reaction-diffusion models"""
    def __init__(self, model, visualization_callback=None):
        self.model = model
        self.visualization_callback = visualization_callback
    
    def apply_boundary_conditions(self, u, v):
        """Apply zero-flux (Neumann) boundary conditions by setting ghost points equal to their neighbors"""
        # Left boundary: u[-1] = u[1] -> u[0] = u[1]
        u[0] = u[1]
        v[0] = v[1]
        
        # Right boundary: u[N+1] = u[N-1] -> u[N] = u[N-1]
        u[-1] = u[-2]
        v[-1] = v[-2]
        
        return u, v
    
    def explicit_euler_step(self, u, v, dt, rhs, boundary_cond=False):
        """Perform one explicit Euler step"""
        #apply boundary conditions
        if boundary_cond == True:
            u, v = self.apply_boundary_conditions(u, v)

        rhs_u, rhs_v = rhs(u, v)
        
        # Update interior points
        u_new = u + dt * rhs_u
        v_new = v + dt * rhs_v

        return u_new, v_new
    
    def heun_step(self, u, v, dt, rhs, boundary_cond=False):
        """Perform one heun step"""
        #apply boundary conditions
        if boundary_cond == True:
            u, v = self.apply_boundary_conditions(u, v)

        # intermediate step
        rhs_u, rhs_v = rhs(u,v)
        u_inter = u + dt * rhs_u
        v_inter = v + dt * rhs_v

        #apply boundary conditions
        if boundary_cond == True:
            u_inter, v_inter = self.apply_boundary_conditions(u_inter, v_inter)

        #update step
        rhs_u_inter, rhs_v_inter = rhs(u_inter,v_inter)
        u_new= u+ dt/2 * (rhs_u + rhs_u_inter)
        v_new = v + dt/2 * (rhs_v + rhs_v_inter)

        return u_new, v_new
        
    def implicit_euler_step(self, u, v):
        """Perform one implicit Euler step"""
        sysmat_u, sysmat_v = self.model.system_matrices_ie
        rhs_u = np.zeros_like(u)
        rhs_v = np.zeros_like(v)
        rhs_u[1:-1] = u[1:-1]
        rhs_v[1:-1] = v[1:-1]
        u_new = sparse.linalg.spsolve(sysmat_u, rhs_u)
        v_new = sparse.linalg.spsolve(sysmat_v, rhs_v)
        return u_new, v_new

    def crank_nicolson_step(self, u ,v):
        """Perform one Crank-Nicolson step"""
        sysmat_u_imp, sysmat_v_imp = self.model.system_matrices_cn_implicit
        sysmat_u_exp, sysmat_v_exp = self.model.system_matrices_cn_explicit
        rhs_u = sysmat_u_exp.dot(u)
        rhs_u[0] = 0
        rhs_u[-1] = 0
        rhs_v = sysmat_v_exp.dot(v)
        rhs_v[0] = 0
        rhs_v[-1] = 0
        u_new = sparse.linalg.spsolve(sysmat_u_imp, rhs_u)
        v_new = sparse.linalg.spsolve(sysmat_v_imp, rhs_v)
        return u_new, v_new
        
    def strang_EE_IE_step(self, u, v):
        """Perform one Strang splitting step"""
        # First half step with explicit Euler (reaction only)
        u_half, v_half = self.explicit_euler_step(u, v, self.model.grid.dt/2, self.model.reaction_terms)
        
        # Full step with implicit Euler (diffusion only)
        u_implicit, v_implicit = self.implicit_euler_step(u_half, v_half)
        
        # Second half step with explicit Euler (reaction only)
        u_new, v_new = self.explicit_euler_step(u_implicit, v_implicit, self.model.grid.dt/2, self.model.reaction_terms)
        return u_new, v_new

    def strang_H_CN_step(self, u, v):
        """Perform one Strang splitting step"""
        # First half step with explicit Euler
        u_half, v_half = self.heun_step(u, v, self.model.grid.dt/2, self.model.reaction_terms)
        
        # Full step with implicit Euler
        u_implicit, v_implicit = self.crank_nicolson_step(u_half, v_half)
        
        # Second half step with explicit Euler
        u_new, v_new = self.heun_step(u_implicit, v_implicit, self.model.grid.dt/2, self.model.reaction_terms)
        
        return u_new, v_new
        
    def solve(self, u0, v0, videomode, method):
        """Solve the PDE system using the specified method"""
        u = u0.copy()
        v = v0.copy()
        
        # Choose the time-stepping method
        step_method = {
            "EE": lambda u, v: self.explicit_euler_step(u, v, self.model.grid.dt, self.model.rhs, boundary_cond=True),
            "IE": self.implicit_euler_step,
            "H": lambda u, v: self.heun_step(u, v, self.model.grid.dt, self.model.rhs, boundary_cond=True),
            "strang_EE_IE": self.strang_EE_IE_step,
            "strang_H_CN": self.strang_H_CN_step
        }[method]
        
        start_time = time.time()
        
        check_interval = 100
        tolerance = 1e-12
        for n in range(self.model.grid.nt):
            
            #compute next iteration
            u_old = u.copy()
            v_old = v.copy()
            u, v = step_method(u, v)

            # Check for steady state every check_interval iterations
            if n % check_interval == 0:
                change = np.max(np.abs(u - u_old))
                if change < tolerance:
                    print(f"\nReached steady state after {n} iterations")
                    self.visualization_callback(u, v, n, override=True)
                    break
            print(f"\rTime step {n+1}/{self.model.grid.nt}       change = {change:.2e}", end=" ", flush=True)

            # visualization
            if videomode:
                self.visualization_callback(u, v, n)
            # handle max. iterations reached
            if n==self.model.grid.nt-1:
                print(f"\nSteady state not reached within {n} iterations")
                self.visualization_callback(u, v, n, override=True)
                
        print(f"Computation completed in {time.time() - start_time:.2f} seconds")
        
        return u, v

def initialize_solution(nx, method):
    """Initialize the solution arrays"""
    if method == "white-noise":
        return np.random.rand(nx), np.random.rand(nx)
    elif method == "spike":
        u0 = np.zeros(nx)
        v0 = np.zeros(nx)
        u0[nx//2] = 1
        v0[nx//2] = 1
        return u0, v0
    else:
        raise ValueError(f"Initialization method {method} not implemented")

def setup_output_directory(dirname):
    """Set up output directory structure"""
    if not os.path.exists(f"out/{dirname}"):
        os.makedirs(f"out/{dirname}/plots")
        os.makedirs(f"out/{dirname}/data")
    else:
        shutil.rmtree(f"out/{dirname}")
        print(f"Old output directory '{dirname}' deleted")
        os.makedirs(f"out/{dirname}/plots")
        os.makedirs(f"out/{dirname}/data")

def create_visualization_callback(model, output_dir, plot_frequency=250):
    """Create a callback function for visualization during solving"""
    def callback(u, v, step, override=False):
        if step % plot_frequency == 0 or override:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(model.grid.x, u, color="purple", label=model.labels["u_label"])
            ax.plot(model.grid.x, v, color="green", label=model.labels["v_label"])
            ax.set_ylim(0, max(np.max(u), np.max(v)))
            ax.set_xlabel(model.labels['x_label'])
            ax.set_ylabel(model.labels['y_label'])
            ax.set_title(f"{model.labels['title']} at t={step*model.grid.dt:.2f}")
            ax.legend()
            fig.savefig(f"out/{output_dir}/plots/solution_{step}")
            plt.close()
            
            # Save data
            np.save(f"out/{output_dir}/data/u_{step}.npy", u)
            np.save(f"out/{output_dir}/data/v_{step}.npy", v)
    
    return callback

def compute_solution(dt, outdir, initialization, videomode, model, timedisc, dimensionless):
    #setup output directory
    setup_output_directory(outdir)

    # Read model parameters and setup grid parameters
    params = ModelParameters(model, settings.read_parameters(), dimensionless)
    params.print()
    grid = GridParameters(
        dimensionless,
        params,
        {"x_start":0,
        "x_end":100,
        "t_start":0,
        "t_end":10,
        "dx":1,
        "dt":dt}
    )
    grid.print()
    
    # Select model
    if model == "NL":
        if dimensionless:
            model = DimensionlessNodalLeftyModel(params,grid)
        else:
            model = NodalLeftyModel(params, grid)
    elif model == "GM":
        model = GiererMeinhardtModel(params, grid)
    
    # Create visualization callback
    vis_callback = create_visualization_callback(
        model,
        outdir,
        plot_frequency=grid.nt//10
    )
    
    # Initialize solver
    solver = TimeStepper(model, visualization_callback=vis_callback)
    
    # Set initial conditions
    np.random.seed(0)
    u0, v0 = initialize_solution(grid.nx, method=initialization)
    
    # Solve system
    u_final, v_final = solver.solve(u0, v0, videomode, timedisc)
    
    return u_final, v_final

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
    #read command line arguments
    args = settings.read_cmdline_args()
    compute_solution(1e-3, args.outdir, args.initialization, args.videomode, args.model, args.timedisc, args.dimensionless)
    # convergence_test(args)