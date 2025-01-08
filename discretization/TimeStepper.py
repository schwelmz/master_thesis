import scipy.sparse as sparse
import numpy as np
import time

class TimeStepper:
    """Handles time stepping for reaction-diffusion models"""
    def __init__(self, model, visualization_callback=None):
        self.model = model
        self.visualization_callback = visualization_callback
    
    def apply_boundary_conditions(self, u, v):
        """Apply zero-flux (Neumann) boundary conditions by setting ghost points equal to their neighbors"""
        if self.model.grid.spatial_dims == 1:
            # Left boundary: u[-1] = u[1] -> u[0] = u[1]
            u[0] = u[1]
            v[0] = v[1]
            
            # Right boundary: u[N+1] = u[N-1] -> u[N] = u[N-1]
            u[-1] = u[-2]
            v[-1] = v[-2]
        if self.model.grid.spatial_dims == 2:
            #top boundary
            u[0,:] = u[1,:]
            v[0,:] = v[1,:]

            #left boundary
            u[:,0] = u[:,1]
            v[:,0] = v[:,1]

            #bottom boundary
            u[-1,:] = u[-2,:]
            v[-1,:] = v[-2,:]

            #right boundary
            u[:,-1] = u[:,-2]
            v[:,-1] = v[:,-2]
        
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
        tolerance = 1e-6
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
