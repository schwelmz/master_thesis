import numpy as np
import settings.settings as settings
import models.ReactionDiffusionModel as models
import models.ModelParameters as ModelParameters
from visualization.visualization import create_visualization_callback
from visualization.visualization import create_visualization_callback_2d
import discretization.TimeStepper as TimeStepper
import discretization.GridParameters as GridParameters

def initialize_solution(shape, method):
    """Initialize the solution arrays"""
    if method == "white-noise":
        if len(shape) == 1:
            return np.random.rand(shape), np.random.rand(shape)
        if len(shape) == 2:
            return np.random.rand(shape[0],shape[1]), np.random.rand(shape[0],shape[1])
    elif method == "spike":
        u0 = np.zeros(shape)
        v0 = np.zeros(shape)
        if len(shape) == 1:
            nx = shape
            u0[nx//2] = 1
            v0[nx//2] = 1
        if len(shape) == 2:
            nx, ny = shape
            u0[nx//2,ny//2] = 1
            v0[nx//2,ny//2] = 1
        return u0, v0
    else:
        raise ValueError(f"Initialization method {method} not implemented")

def compute_solution(dt, outdir, initialization, videomode, model, timedisc, dimensionless, spatial_dims):
    #setup output directory
    settings.setup_output_directory(outdir)

    # Read model parameters and setup grid parameters
    params = ModelParameters.ModelParameters(model, settings.read_parameters(), dimensionless)
    params.print()
    grid = GridParameters.GridParameters(
        dimensionless,
        spatial_dims,
        params,
        {"x_start":0,
        "y_start":0,
        "x_end":100,
        "y_end":100,
        "t_start":0,
        "t_end":100,
        "dx":1,
        "dy":1,
        "dt":dt}
    )
    grid.print()
    
    # Select model
    if model == "NL":
        if dimensionless:
            model = models.DimensionlessNodalLeftyModel(params,grid)
        else:
            model = models.NodalLeftyModel(params, grid)
    elif model == "GM":
        model = models.GiererMeinhardtModel(params, grid)
    
    # Create visualization callback
    if spatial_dims == 1:
        vis_callback = create_visualization_callback(
            model,
            outdir,
            plot_frequency=grid.nt//10
        )
    elif spatial_dims == 2:
        vis_callback = create_visualization_callback_2d(
            model,
            outdir,
            plot_frequency=grid.nt//10
        )
    
    # Initialize solver
    solver = TimeStepper.TimeStepper(model, visualization_callback=vis_callback)
    
    # Set initial conditions
    np.random.seed(0)
    u0, v0 = initialize_solution((grid.nx, grid.ny), method=initialization)
    
    # Solve system
    u_final, v_final = solver.solve(u0, v0, videomode, timedisc)
    
    return u_final, v_final

if __name__ == "__main__":
    args = settings.read_cmdline_args()
    compute_solution(1e-3, args.outdir, args.initialization, args.videomode, args.model, args.timedisc, args.dimensionless, int(args.dimensions))