import matplotlib.pyplot as plt
import numpy as np
import imageio
import matplotlib.colors as clr

# Define the colors
black_orange = ["black", "salmon"]  # RGB values for black and orange
black_blue = ["black", "cornflowerblue"]  # RGB values for black and blue
# Create the colormap
cm_bo = clr.LinearSegmentedColormap.from_list("black_orange", black_orange)
cm_bb = clr.LinearSegmentedColormap.from_list("black_blue", black_blue)

def heatmap(fig,axs,N,L,timestep,extent,time):
    fig.suptitle(f"t = {time//60:.0f}hrs {time%60:.2f}min")
    #ax0
    im0 = axs[0].imshow(N[:,:],extent=extent,origin="lower",vmin=0,cmap=cm_bo)
    axs[0].set_title(f"Species A")
    axs[0].set_xlabel(r"domain width [$\mu m$]")
    axs[0].set_ylabel(r"domain height [$\mu m$]")
    cb0 = fig.colorbar(im0, ax=axs[0])
    cb0.set_label(r"Nodal [$nM/\mu m$]")
    #ax1
    im1 = axs[1].imshow(L[:,:],extent=extent,origin="lower",vmin=0,cmap=cm_bb)
    axs[1].set_title(f"Species B")
    axs[1].set_xlabel(r"domain width [$\mu m$]")
    axs[1].set_ylabel(r"domain height [$\mu m$]")
    cb1 = fig.colorbar(im1, ax=axs[1])
    cb1.set_label(r"Lefty [nM/\mu m]")


def create_gif(N,L,extent,Nt,ht,skips):
    fig,axs = plt.subplots(1,2,figsize=(12,5))
    for tdx in range(0,Nt,skips):
        print("\r",tdx,end="",flush=True)
        fig.suptitle(f"t={tdx*ht:.5f}min")
        #ax0
        im0 = axs[0].imshow(N[tdx,:,:],extent=extent,origin="lower")
        axs[0].set_title(f"Nodal")
        axs[0].set_xlabel(r"domain width [$\mu m$]")
        axs[0].set_ylabel(r"domain height [$\mu m$]")
        cb0 = fig.colorbar(im0, ax=axs[0])
        cb0.set_label("Nodal [nM]")
        #ax1
        im1 = axs[1].imshow(L[tdx,:,:],extent=extent,origin="lower")
        axs[1].set_title(f"Lefty")
        axs[1].set_xlabel(r"domain width [$\mu m$]")
        axs[1].set_ylabel(r"domain height [$\mu m$]")
        cb1 = fig.colorbar(im1, ax=axs[1])
        cb1.set_label("Lefty [nM]")
        #savefig
        fig.savefig('plots/heatmap_frame_'+str(tdx)+'.png')
        cb0.remove()
        cb1.remove()
        axs[0].cla()
        axs[1].cla()
    # Build GIF
    print('\n building .gif file')
    [xstart, xend, ystart, yend] = extent
    with imageio.get_writer(f'heatmap_neumann_{Nt}_{skips}_{xend}_{yend}.mp4', fps=20) as writer:
        for filename in range(0,Nt,skips):
            image = imageio.v2.imread('plots/heatmap_frame_' + str(filename)+ '.png')
            writer.append_data(image)