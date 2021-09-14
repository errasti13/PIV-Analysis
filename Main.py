import numpy as np
import math
import tkinter as tk
from tkinter import filedialog as fd
from normxcorr2 import normxcorr2
import cv2 as cv2
from data_validation import st_dev_check
from Subpixel_fit import gaussian_fit, parabolic_fit
from Imagenes_sinteticas import flujo_poiseuille, flujo_couette,desplazamiento_horizontal

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['k'])
import matplotlib.pyplot as plt

A = 1024
n = np.int64(0.05*(A**2))
dt= 1 #Diametro de la partícula
q=255
sigma=1/8.*(dt**2)
delta_z0=3
delta_z=1
H =int(30/100)
if delta_z<delta_z0:
    I0=q
else:
    I0=0
    
print("1. Crear imágenes sintéticas")
print("2. Analizar imágenes")

numero =np.int32(input())

if numero < 2:
    print("Seleccione un campo de velocidades")
    print("1.Flujo de Poiseuille")
    print("2.Flujo de Couette")
    print("3.Desplazamiento horizontal")
    numero2 = np.int32(input())
    if 0<numero2<2:
        ## Campo de velocidades
        H = 30/100 # diámetro del tubo en ms
        I = flujo_poiseuille(A, n, H, dt, I0)
        
    if 1<numero2<3:
        ## Campo de velocidades
        H = 30/100 # diámetro del tubo en ms
        U = 10
        [I1,I2] = flujo_couette(A, n, H, dt, I0, U)
        
    if 2<numero2<4:
       I = desplazamiento_horizontal(A, n, dt, I0)

else:
        root = tk.Tk()
        filez = np.array(fd.askopenfilenames(parent=root,title='Choose a file'))
        I = []
        for i in filez:
            I.append(cv2.imread(i,0))


if numero <2:
    num_imagenes = 2
else:
    num_imagenes = len(I)

#Window size
print("Introduzca el tamaño de las ventanas de interrogación")
w_width =np.int8(input())
w_height = w_width

print("Indique el estimador sub píxel que desee utilizar")
print("1. Ajuste de Gauss con tres puntos")
print("2. Ajuste parabólico")
sbpx_method =np.int8(input())
for i in range (num_imagenes-1):
    I1 = I[i]
    I2 = I[i+1]
    
    xmax = I1.shape[0]
    ymax = I1.shape[1]

    
    #Centre of grid points
    x_min = w_width/2.
    y_min = w_height/2.
    xgrid = np.arange(x_min,xmax-w_width/2.,w_width/2.)
    ygrid = np.arange(y_min,(ymax-w_height), w_height/2.)
    
    #Number of windwos
    x_count = xgrid.shape[0]
    y_count = ygrid.shape[0]
    
    #Ranges of search zones
    x_disp_max = np.int8(w_width/2.)
    y_disp_max = np.int8(w_height/2.)
    
    #Hay que crear las ventanas de interrogación y búsqueda
    test_ima = np.zeros([w_width,w_height])
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
       
    [dpx,dpy,G] = [np.nan_to_num(dpx),np.nan_to_num(dpy),np.nan_to_num(G)]
    
    [dpx2,dpy2,G2] = st_dev_check(dpx,dpy,G,5,5,5)
      
    [dpx2,dpy2,G2] = [np.float32(dpx2), np.float32(dpy2), np.float32(G2)]
    
    dpx_smooth = cv2.medianBlur(dpx2,3)
    dpy_smooth = cv2.medianBlur(dpy2,3)
    G_smooth = cv2.medianBlur(G2,3)
    
    theta = np.zeros([dpx.shape[0],dpx.shape[1]])
    for i in range (0,dpx.shape[0]):
        for j in range(0,dpx.shape[1]):
            theta[i,j] = np.arctan(-dpx_smooth[i,j]/dpy_smooth[i,j])*180/np.pi
            
    theta = np.nan_to_num(theta)
    theta = np.float32(theta)
    theta = cv2.medianBlur(theta,3)
    
    #Cambio de pixel/frame a m/s
   # pixel = (54*10**(-3))/(1830-175)
   # dpx_smooth = dpx_smooth*pixel*1000
   # dpy_smooth = dpy_smooth*pixel*1000
   # G_smooth = G_smooth*pixel*1000
    
    #Dibuja cosas
    plt.figure()
    plt.quiver(-dpx_smooth, dpy_smooth, color = 'Green')
    plt.title("Campo de velocidades")
    #plt.savefig('Imagenes/Solución_final.eps', format='eps')
        
    plt.figure()
    plt.plot(dpy_smooth,-dpx_smooth,'.', color = 'black')
    plt.xlabel("Desplazamientos en X (píxel/frame)")
    plt.ylabel("Desplazamientos en Y (píxel/frame)")
   # plt.savefig('Imagenes/MapaUV.eps', format='eps')
    
    plt.figure()
    plt.imshow(dpy_smooth, interpolation = 'bilinear')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("Desplazamientos según el eje X (píxel/frame)")
    plt.savefig('Imagenes 64/MapaCalorU.eps', format='eps')
    
    plt.figure()
    plt.imshow(-dpx_smooth, interpolation = 'bilinear')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("Desplazamientos según el eje y")
   # plt.savefig('Imagenes/MapaCalorV.eps', format='eps')
    
    plt.figure()
    plt.imshow(G_smooth,interpolation = 'bilinear')#,extent = [-4.5,3.9,0,6.3])
    plt.colorbar()
    plt.title("Módulo de la velocidad")
   # plt.savefig('Imagenes/MapaCalorMódulo.eps', format='eps')
    
    plt.figure()
    plt.imshow(theta, interpolation = 'bilinear')#,extent = [-4.5,3.9,0,6.3])
    plt.title("Ángulo que forma el vector con la vertical")
    plt.colorbar()
    #plt.savefig('Imagenes/MapaCalorTheta.eps', format='eps')
    
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
    plt.savefig("Imagenes 64/ValorU.eps", format ='eps')
    