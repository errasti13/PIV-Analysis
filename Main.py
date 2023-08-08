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
    
def plot_results(dpx_smooth, dpy_smooth, G_smooth):
        #Dibuja cosas
    plt.figure()
    plt.quiver(dpy_smooth, -dpx_smooth, color = 'Green')
    plt.title("Campo de velocidades")
    #plt.savefig('Imagenes/Solución_final.eps', format='eps')
        
    plt.figure()
    plt.plot(dpy_smooth,dpx_smooth,'.', color = 'black')
    plt.xlabel("Desplazamientos en X (píxel/frame)")
    plt.ylabel("Desplazamientos en Y (píxel/frame)")
   # plt.savefig('Imagenes/MapaUV.eps', format='eps')
    
    plt.figure()
    plt.imshow(dpy_smooth, interpolation = 'bilinear')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("Desplazamientos según el eje X (píxel/frame)")
    #plt.savefig('Imagenes 64/MapaCalorU.eps', format='eps')
    
    plt.figure()
    plt.imshow(dpx_smooth, interpolation = 'bilinear')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("Desplazamientos según el eje y")
   # plt.savefig('Imagenes/MapaCalorV.eps', format='eps')
    
    plt.figure()
    plt.imshow(G_smooth,interpolation = 'bilinear')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("Módulo de la velocidad")
   # plt.savefig('Imagenes/MapaCalorMódulo.eps', format='eps')
    
    plt.figure()
    plt.hist(dpy_smooth, histtype = 'stepfilled',bins = 100)
    plt.xlabel("Valor del desplazamiento en el eje X")
    plt.ylabel("Frecuencia")
    #plt.savefig('Imagenes/HistogramaU.eps', format='eps')
    
    
    plt.figure()
    plt.hist(dpx_smooth, histtype = 'stepfilled',bins = 100)
    plt.xlabel("Valor del desplazamiento en el eje Y")
    plt.ylabel("Frecuencia")
   # plt.savefig('Imagenes/HistogramaV.eps', format='eps')
    
    plt.figure()
    plt.hist(G_smooth, histtype = 'stepfilled',bins = 100)
    plt.xlabel("Módulo del desplazamiento")
    plt.ylabel("Frecuencia")
   # plt.savefig('Imagenes/HistogramaG.eps', format='eps')
   
    plt.figure()
    #x_values = np.linspace(-4.5,3.9,157)
    plt.plot(dpy_smooth[30,:])
    plt.title("Componente U de la velocidad a lo largo del eje X")
    plt.xlabel("Posición en X")
    plt.ylabel("Valor de U (pixel/frame)")
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
       
    [dpx,dpy,G] = [np.nan_to_num(dpx),np.nan_to_num(dpy),np.nan_to_num(G)]
    
    [dpx2,dpy2,G2] = st_dev_check(dpx,dpy,G,5,5,5)
      
    [dpx2,dpy2,G2] = [np.float32(dpx2), np.float32(dpy2), np.float32(G2)]
    
    dpx_smooth = dpx #cv2.medianBlur(dpx2,3)
    dpy_smooth = dpy#cv2.medianBlur(dpy2,3)
    G_smooth = G #cv2.medianBlur(G2,3)

    plot_results(dpx_smooth, dpy_smooth, G_smooth)
    


if __name__ == "__main__":
    main()
    
