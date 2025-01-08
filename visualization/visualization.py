import matplotlib.pyplot as plt
import numpy as np

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

def create_visualization_callback_2d(model, output_dir, plot_frequency=250):
    """Create a callback function for 2D visualization during solving"""
    def callback(u, v, step, override=False):
        if step % plot_frequency == 0 or override:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = ax1.imshow(u.T, origin='lower', extent=[model.grid.x_start, model.grid.x_end, 
                                                         model.grid.y_start, model.grid.y_end], 
                                                         cmap=plt.get_cmap("BuPu"),
                                                         vmin=0)
            ax1.set_xlabel(model.labels["x_label"])
            ax1.set_ylabel(model.labels["y_label"])
            cb_u = plt.colorbar(im1, ax=ax1)
            cb_u.set_label(model.labels["u_label"])
            
            im2 = ax2.imshow(v.T, origin='lower', extent=[model.grid.x_start, model.grid.x_end,
                                                         model.grid.y_start, model.grid.y_end], 
                                                         cmap=plt.get_cmap("PuBuGn"),
                                                         vmin=0)
            ax2.set_xlabel(model.labels["x_label"])
            ax2.set_ylabel(model.labels["y_label"])
            cb_v = plt.colorbar(im2, ax=ax2)
            cb_v.set_label(model.labels["v_label"])

            fig.suptitle(f"t = {step*model.grid.dt:.2f}")
            
            fig.savefig(f"out/{output_dir}/plots/solution_{step}")
            plt.close()
            
            # Save data
            np.save(f"out/{output_dir}/data/u_{step}.npy", u)
            np.save(f"out/{output_dir}/data/v_{step}.npy", v)
    
    return callback
