import numpy as np

class GridParameters:
    """Container for spatial and temporal grid parameters"""
    def __init__(self, is_dimensionless, spatial_dims, model_params, grid_params_dict):
        self.spatial_dims = spatial_dims
        if is_dimensionless:
            self._init_dimensionless(grid_params_dict)
        else:
            self._init_dimensional(model_params, grid_params_dict)
    
    def _init_dimensionless(self, grid_params_dict):
        #spatial discretization in x-direction
        self.x_start = grid_params_dict["x_start"]
        self.x_end = grid_params_dict["x_end"]
        self.dx = grid_params_dict["dx"]
        self.nx = int((self.x_end - self.x_start) / self.dx)
        self.x = np.linspace(self.x_start, self.x_end, self.nx)
        #spatial discretization in y-direction
        if self.spatial_dims == 2:
            self.y_start = grid_params_dict["y_start"]
            self.y_end = grid_params_dict["y_end"]
            self.dy = grid_params_dict["dy"]
            self.ny = int((self.y_end - self.y_start) / self.dy)
            self.y = np.linspace(self.y_start, self.y_end, self.ny)
        #temporal discretization
        self.t_start = grid_params_dict["t_start"]
        self.t_end = grid_params_dict["t_end"]
        self.dt = grid_params_dict["dt"]
        self.nt = int((self.t_end - self.t_start) / self.dt)
    
    def _init_dimensional(self, model_params, grid_params_dict):
        scale_factor = np.sqrt(model_params.D_N/model_params.gamma_N)
        #spatial discretization in x-direction
        self.x_start = grid_params_dict["x_start"]
        self.x_end = grid_params_dict["x_end"] * scale_factor
        self.dx = grid_params_dict["dx"]
        self.nx = int((self.x_end - self.x_start) / self.dx)
        self.x = np.linspace(self.x_start, self.x_end, self.nx)
        #spatial discretization in y-direction
        if self.spatial_dims == 2:
            self.y_start = grid_params_dict["y_start"]
            self.y_end = grid_params_dict["y_end"] * scale_factor
            self.dy = grid_params_dict["dy"]
            self.ny = int((self.y_end - self.y_start) / self.dy)
            self.y = np.linspace(self.y_start, self.y_end, self.ny)
        #temporal discretization
        self.t_start = grid_params_dict["t_start"]
        self.t_end = grid_params_dict["t_end"] / model_params.gamma_N
        self.dt = grid_params_dict["dt"]
        self.nt = int((self.t_end - self.t_start) / self.dt)

    def print(self):
        print('Grid parameters:')
        print('\n'.join("\t%s: %s" % (key,value) for key,value in vars(self).items() if (key != "x" and key != "y")))