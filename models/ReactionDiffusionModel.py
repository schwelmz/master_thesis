from functools import lru_cache
import scipy.sparse as sparse
import numpy as np

class ReactionDiffusionModel:
    """Base class for reaction-diffusion models"""
    def __init__(self, params, grid):
        self.params = params
        self.grid = grid
        self._setup_system_matrices()
        
    @property
    def labels(self):
        """Return dictionary of labels for plotting"""
        if self.grid.spatial_dims == 1:
            return {
                'x_label': 'x',
                'y_label': 'Concentration',
                'title': 'Reaction-Diffusion System',
                'u_label': 'Species U',
                'v_label': 'Species V'
            }
        elif self.grid.spatial_dims == 2:
            return {
                'x_label': 'x',
                'y_label': 'y',
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

    def _laplacian(self, U):
        """Calculate Laplacian using central differences"""
        if self.grid.spatial_dims == 1:
            return (U[0:-2] - 2*U[1:-1] + U[2:]) / (self.grid.dx ** 2)
        elif self.grid.spatial_dims == 2:
            d2x = (U[0:-2,1:-1] + U[2:,1:-1] - 2*U[1:-1,1:-1]) / (self.grid.dx**2)
            d2y = (U[1:-1,0:-2] + U[1:-1,2:] - 2*U[1:-1,1:-1]) / (self.grid.dy**2)
            return d2x + d2y

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
        if self.grid.spatial_dims == 1:
            return {
                'x_label': r'Position ($\mu m$)',
                'y_label': r'Concentration ($nM$)',
                'title': 'Nodal-Lefty System',
                'u_label': 'Nodal',
                'v_label': 'Lefty'
        }
        elif self.grid.spatial_dims == 2:
            return {
                'x_label': r'x [$\mu m$]',
                'y_label': r'y [$\mu m$]',
                'title': 'Nodal-Lefty System',
                'u_label': r'Nodal [$nM$]',
                'v_label': r'Lefty [$nM$]'
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
        rhs_N = np.zeros_like(N)
        rhs_L = np.zeros_like(L)
        if self.grid.spatial_dims == 1:
            r_N, r_L = self.reaction_terms(N[1:-1], L[1:-1])
            rhs_N[1:-1] = r_N + self.params.D_N * diff_N
            rhs_L[1:-1] = r_L + self.params.D_L * diff_L
        elif self.grid.spatial_dims == 2:
            r_N, r_L = self.reaction_terms(N[1:-1,1:-1], L[1:-1,1:-1])
            rhs_N[1:-1,1:-1] = r_N + self.params.D_N * diff_N
            rhs_L[1:-1,1:-1] = r_L + self.params.D_L * diff_L
        return rhs_N, rhs_L
    
class DimensionlessNodalLeftyModel(ReactionDiffusionModel):
    """Implementation of the dimensionless Nodal-Lefty model"""
    @property
    def labels(self):
        """Return dictionary of labels for plotting"""
        if self.grid.spatial_dims == 1:
            return {
                'x_label': r'Dimensionless Position $x^*$',
                'y_label': 'Dimensionless Concentration',
                'title': 'Dimensionless Nodal-Lefty System',
                'u_label': 'Nodal',
                'v_label': 'Lefty'
            }
        elif self.grid.spatial_dims == 2:
            return {
                'x_label': r'Dimensionless Position $x^*$',
                'y_label': r'Dimensionless Position $y^*$',
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
        rhs_N = np.zeros_like(N)
        rhs_L = np.zeros_like(L)
        if self.grid.spatial_dims == 1:
            r_N, r_L = self.reaction_terms(N[1:-1], L[1:-1])
            rhs_N[1:-1] = r_N + diff_N
            rhs_L[1:-1] = r_L + self.params.d * diff_L
        elif self.grid.spatial_dims == 2:
            r_N, r_L = self.reaction_terms(N[1:-1,1:-1], L[1:-1,1:-1])
            rhs_N[1:-1,1:-1] = r_N + diff_N
            rhs_L[1:-1,1:-1] = r_L + self.params.d * diff_L
        
        return rhs_N, rhs_L

class GiererMeinhardtModel(ReactionDiffusionModel):
    """Implementation of the Gierer-Meinhardt model"""
    @property
    def labels(self):
        """Return dictionary of labels for plotting"""
        if self.grid.spatial_dims == 1:
            return {
                'x_label': 'Position x',
                'y_label': 'Concentration',
                'title': 'Gierer-Meinhardt System',
                'u_label': 'Species A',
                'v_label': 'Species B'
            }
        elif self.grid.spatial_dims == 2:
            return {
                'x_label': 'Position x',
                'y_label': 'Position y',
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
        rhs_U = np.zeros_like(U)
        rhs_V = np.zeros_like(V)
        if self.grid.spatial_dims == 1:
            r_U, r_V = self.reaction_terms(U[1:-1], V[1:-1])
            rhs_U[1:-1] = r_U + self.params.D_u * diff_U
            rhs_V[1:-1] = r_V + self.params.D_v * diff_V
        elif self.grid.spatial_dims == 2:
            r_U, r_V = self.reaction_terms(U[1:-1,1:-1], V[1:-1,1:-1])
            rhs_U[1:-1,1:-1] = r_U + self.params.D_u * diff_U
            rhs_V[1:-1,1:-1] = r_V + self.params.D_v * diff_V
        
        return rhs_U, rhs_V