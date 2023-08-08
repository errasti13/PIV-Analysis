"""
Particle Image Velocimetry (PIV) analysis script.

This script performs displacement analysis on a sequence of grayscale images using normalized cross-correlation.
It calculates sub-pixel accurate displacements between consecutive frames, generating gradient magnitude values.

Author: Jon Errasti Odriozola
Date: September 2021
Last modified: August 2023
"""

# Import necessary libraries
import numpy as np
import math
import cv2 as cv2
import tkinter as tk
from tkinter import filedialog as fd

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['k'])
import matplotlib.pyplot as plt

from normxcorr2 import normxcorr2
from data_validation import st_dev_check
from Subpixel_fit import gaussian_fit, parabolic_fit


def load_images(file_paths):
    """
    Load and normalize grayscale images from the specified file paths.

    Args:
        file_paths (list): List of paths to image files.

    Returns:
        numpy.ndarray: Array containing loaded and normalized images.
    """
    images = []
    for path in file_paths:
        image = cv2.imread(path, 0) / 255
        images.append(image)

    return np.asarray(images)

def compute_displacement(image_sequence, window_size=64, sbpx_method=1):
    """
    Compute sub-pixel accurate displacements between consecutive images in a sequence.

    Args:
        image_sequence (numpy.ndarray): Array of grayscale images.
        window_size (int): Size of the correlation window.
        sbpx_method (int): Subpixel adjustment method (1 for Gaussian, 2 for parabolic).

    Returns:
        numpy.ndarray: Array of calculated displacements between images.
    """
    displacements = []

    w_width =window_size
    w_height = w_width

    for i in range(image_sequence.shape[0] - 1):
        I1 = image_sequence[i]
        I2 = image_sequence[i + 1]
        
        xmax = I1.shape[0]
        ymax = I1.shape[1]

        
        # Define grid points for windows
        x_min = w_width/2.
        y_min = w_height/2.
        xgrid = np.arange(x_min,xmax-w_width/2.,w_width/2.)
        ygrid = np.arange(y_min,(ymax-w_height), w_height/2.)
        
        x_count = xgrid.shape[0] # Number of windows in x-direction
        y_count = ygrid.shape[0] # Number of windows in y-direction
        
        x_disp_max = np.int8(w_width/2.) # Maximum x-displacement within a window
        y_disp_max = np.int8(w_height/2.) # Maximum y-displacement within a window
        
        test_ima = np.zeros([w_width,w_height]) # Placeholder for a test image
        test_imb = np.zeros([w_width+2*x_disp_max,w_height+2*y_disp_max])
        dpx = np.zeros([x_count,y_count])
        dpy = np.zeros([x_count,y_count])
        xpeak = 0
        ypeak = 0
        xpeak1 = 0
        ypeak1 = 0
        xpeak_sbpx = 0
        ypeak_sbpx = 0

        G = np.zeros([x_count,y_count])
        for i in range (1,x_count,1):
            for j in range (1,y_count,1):
                max_correlation = 0
                test_xmin = np.int32(xgrid[i]-w_width/2.)
                test_xmax = np.int32(xgrid[i]+w_width/2.)
                test_ymin = np.int32(ygrid[j]-w_height/2.)
                test_ymax = np.int32(ygrid[j]+w_height/2.)
                x_disp = 0
                y_disp = 0 
                test_ima = I1[test_xmin:test_xmax,test_ymin:test_ymax]
                test_imb = I2[(test_xmin-x_disp_max):(test_xmax+x_disp_max),(test_ymin-y_disp_max):(test_ymax+y_disp_max)]
                c = normxcorr2(test_ima,test_imb)
                xpeak,ypeak = np.unravel_index(c.argmax(),c.shape)
                
                #Utilizamos estimadores sub píxel para mayor precisión
                if sbpx_method<2:
                    [xpeak_sbpx,ypeak_sbpx] = gaussian_fit(xpeak,ypeak,c)
                else:
                    [xpeak_sbpx,ypeak_sbpx] = parabolic_fit(xpeak,ypeak,c)
                
                xpeak1 = test_xmin + xpeak_sbpx - w_width/2. - x_disp_max
                ypeak1 = test_ymin + ypeak_sbpx - w_height/2. - y_disp_max
                dpx[i,j] = xpeak1 - xgrid[i]
                dpy[i,j] = ypeak1 - ygrid[j]
                G[i,j] = math.sqrt((dpx[i,j])**2+(dpy[i,j])**2)

        return dpx, dpy, G
    
def svd_reconstruction(dpx, dpy, G, m):

    """
    Perform Singular Value Decomposition (SVD) based reconstruction and smoothing on input arrays.

    Args:
        dpx (numpy.ndarray): Array of x-displacements.
        dpy (numpy.ndarray): Array of y-displacements.
        G (numpy.ndarray): Array of velocity magnitudes.
        m (int): Number of singular values to consider for smoothing.

    Returns:
        numpy.ndarray: Smoothed and reconstructed arrays of x-displacements, y-displacements, and velocity magnitudes.
    """
    # Check and mask values based on standard deviation
    [dpx, dpy, G] = st_dev_check(dpx, dpy, G, 3, 3, 3)
    
    # Convert arrays to float32 data type
    dpx = np.float32(dpx)
    dpy = np.float32(dpy)
    G = np.float32(G)

    # Replace NaNs with means
    dpx = np.nan_to_num(dpx, nan=np.nanmean(dpx))
    dpy = np.nan_to_num(dpy, nan=np.nanmean(dpy))
    G = np.nan_to_num(G, nan=np.nanmean(G))

    # Perform Singular Value Decomposition (SVD)
    u_dpx, s_dpx, vt_dpx = np.linalg.svd(dpx, full_matrices = False)
    u_dpy, s_dpy, vt_dpy = np.linalg.svd(dpy, full_matrices = False)
    u_G, s_G, vt_G = np.linalg.svd(G, full_matrices = False)

    dpx_smooth = np.zeros(dpx.shape)
    dpy_smooth = np.zeros(dpy.shape)
    G_smooth = np.zeros(G.shape)


    # Smooth all arrays using a single loop
    for i in range(m):
        dpx_smooth += s_dpx[i] * np.outer(u_dpx[:, i], vt_dpx[i, :])
        dpy_smooth += s_dpy[i] * np.outer(u_dpy[:, i], vt_dpy[i, :])
        G_smooth += s_G[i] * np.outer(u_G[:, i], vt_G[i, :])

    return dpx_smooth, dpy_smooth, G_smooth


def plot_results(dpx_smooth, dpy_smooth, G_smooth):

    """
    Generate a series of plots and visualizations to analyze PIV (Particle Image Velocimetry) results.

    This function takes in smoothed displacement data and generates several types of plots and visualizations
    to provide insights into the velocity vector field, displacement magnitudes, and distribution statistics.

    Args:
        dpx_smooth (numpy.ndarray): Smoothed y-component of displacements.
        dpy_smooth (numpy.ndarray): Smoothed x-component of displacements.
        G_smooth (numpy.ndarray): Smoothed magnitude of displacements (velocity magnitudes).

    Returns:
        None (Plots are generated using Matplotlib).
    """

    plt.figure()
    plt.quiver(dpy_smooth, -dpx_smooth, color='lime')
    plt.title("Velocity vector field")
    #plt.savefig('Imagenes/Solución_final.eps', format='eps')
        
    plt.figure()
    plt.plot(dpy_smooth,dpx_smooth,'.', color = 'black')
    plt.xlabel("X displacements (pixel/frame)")
    plt.ylabel("Y displacements (pixel/frame)")
   # plt.savefig('Imagenes/MapaUV.eps', format='eps')
    
    plt.figure()
    plt.imshow(dpy_smooth, interpolation = 'bicubic')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("X displacements (pixel/frame)")
    #plt.savefig('Imagenes 64/MapaCalorU.eps', format='eps')
    
    plt.figure()
    plt.imshow(dpx_smooth, interpolation = 'bicubic')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("Y displacements")
   # plt.savefig('Imagenes/MapaCalorV.eps', format='eps')
    
    plt.figure()
    plt.imshow(G_smooth,interpolation = 'bicubic')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("Velocity magnitude")
   # plt.savefig('Imagenes/MapaCalorMódulo.eps', format='eps')
    
    plt.figure()
    plt.hist(dpy_smooth, histtype = 'stepfilled',bins = 100)
    plt.xlabel("X velocity value")
    plt.ylabel("Frequency")
    #plt.savefig('Imagenes/HistogramaU.eps', format='eps')
    
    
    plt.figure()
    plt.hist(dpx_smooth, histtype = 'stepfilled',bins = 100)
    plt.xlabel("Y velicity value")
    plt.ylabel("Frequency")
   # plt.savefig('Imagenes/HistogramaV.eps', format='eps')
    
    plt.figure()
    plt.hist(G_smooth, histtype = 'stepfilled',bins = 100)
    plt.xlabel("Velocity magnitude")
    plt.ylabel("Frequency")
   # plt.savefig('Imagenes/HistogramaG.eps', format='eps')
   
    plt.figure()
    #x_values = np.linspace(-4.5,3.9,157)
    plt.plot(dpy_smooth[30,:])
    plt.title("X velocity fluctuation")
    plt.xlabel("X Position")
    plt.ylabel("U velocity(pixel/frame)")
    plt.show()
    #plt.savefig("Imagenes 64/ValorU.eps", format ='eps')

def main():
    """
    Main function to load images, compute displacements, and perform further analysis.

    This function serves as the entry point for the script's execution.
    """
    # root = tk.Tk()  # GUI library code that seems to be commented out
    filez = ['/mnt/c/Users/Jon/Downloads/025-1ms/025-1ms/025-1ms_00000600.tif', '/mnt/c/Users/Jon/Downloads/025-1ms/025-1ms/025-1ms_00000601.tif']
    
    # Load grayscale images
    image_sequence = load_images(filez)
    
    # Compute displacements
    [dpx, dpy, G] = compute_displacement(image_sequence)
    
    m = 18
    dpx_smooth, dpy_smooth, G_smooth = svd_reconstruction(dpx,dpy,G,m)

    plot_results(dpx_smooth, dpy_smooth, G_smooth)
    


if __name__ == "__main__":
    main()
    
