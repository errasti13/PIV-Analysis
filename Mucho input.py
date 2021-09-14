import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from mpl_toolkits.mplot3d import axes3d
import math
import tkinter as tk
from tkinter import filedialog as fd
from normxcorr2 import normxcorr2
import cv2 as cv2
from data_validation import st_dev_check

A = 1024
n = np.int64(0.05*(A**2))
dt= 2 #Diametro de la partícula
q=255
sigma=1/8.*(dt**2)
delta_z0=3
delta_z=1
H = 30/100
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
        y = np.linspace(0, H, A+2*dt)
        u = -1/(2*0.0000174)*(y**2-H*y)*2.6458 #Cambio de m/s a pixeles/s
        d = u*0.001
        d_max = np.int64(np.round(np.max(d)+1))
        d_round = np.int64(np.round(d))
        plt.plot(u)
        
        i = 0
        I1=np.zeros([A+2*dt,A+2*dt])
        I2=np.zeros([A+d_max+2*dt,A+2*dt])
        
        for i in range (0,n,1):
              i = i + 1
              x_0 = np.random.random()*A
              y_0 = np.random.random()*A
              x = np.int64(np.round(x_0))
              y = np.int64(np.round(y_0))
              v_x = np.arange(x-2*dt,x+2*dt,1)
              v_y = np.arange(y-2*dt,y+2*dt,1)
              for j in v_y:
                  I1[v_x,j]=I1[v_x,j]+I0*np.exp((-(v_x-x_0)**2-(j-y_0)**2)/sigma)
                  I2[v_x+(d_round[j]),j]= I1[v_x,j]
                
        I1=np.delete(I1, np.s_[(A):(A+2*dt)], axis=0)
        I1=np.delete(I1, np.s_[(A):(A+2*dt)], axis=1)
        I2=np.delete(I2, np.s_[(A):(A+d_max+2*dt)], axis=0)                            
        I2=np.delete(I2, np.s_[(A):(A+2*dt)], axis=1) 
        
    if 1<numero2<3:
        ## Campo de velocidades
        H = 30/100 # diámetro del tubo en ms
        y = np.linspace(0, H, A+2*dt)
        U = 10
        u = (U/H)*y*2.6458 #Cambio de m/s a pixeles/s
        d = u*0.1
        d_max = np.int64(np.round(np.max(d)+1))
        d_round = np.int64(np.round(d))
        plt.plot(u)
        
        i = 0
        I1=np.zeros([A+2*dt,A+2*dt])
        I2=np.zeros([A+d_max+2*dt,A+2*dt])
        
        for i in range (0,n,1):
            i = i + 1
            x_0 = np.random.random()*A
            y_0 = np.random.random()*A
            x = np.int64(np.round(x_0))
            y = np.int64(np.round(y_0))
            v_x = np.arange(x-2*dt,x+2*dt,1)
            v_y = np.arange(y-2*dt,y+2*dt,1)
            for j in v_y:
                I1[v_x,j]=I1[v_x,j]+I0*np.exp((-(v_x-x_0)**2-(j-y_0)**2)/sigma)
                I2[v_x+(d_round[j]),j]= I1[v_x,j]
                
        I1=np.delete(I1, np.s_[(A):(A+2*dt)], axis=0)
        I1=np.delete(I1, np.s_[(A):(A+2*dt)], axis=1)
        I2=np.delete(I2, np.s_[(A):(A+d_max+2*dt)], axis=0)                            
        I2=np.delete(I2, np.s_[(A):(A+2*dt)], axis=1)
        
    if 2<numero2<4:
        print("Número de píxeles que desee desplazar")
        d = np.int32(input())
        i = 0
        I1=np.zeros([A+2*dt,A+2*dt])
        I2=np.zeros([A+d+2*dt,A+2*dt])

        for i in range (0,n,1):
            i = i + 1
            x_0 = np.random.random()*A
            y_0 = np.random.random()*A
            x = np.round(x_0)
            y = np.round(y_0)
            v_x = np.int32(np.arange(x-2*dt,x+2*dt,1))
            v_y = np.int32(np.arange(y-2*dt,y+2*dt,1))
            for j in v_x:
                I1[j,v_y]=I1[j,v_y]+I0*np.exp((-(j-x_0)**2-(v_y-y_0)**2)/sigma)
                I2[j+d,v_y]= I1[j,v_y]

        I1=np.delete(I1, np.s_[(A):(A+2*dt)], axis=0)
        I1=np.delete(I1, np.s_[(A):(A+2*dt)], axis=1)
        I2=np.delete(I2, np.s_[(A):(A+d+2*dt)], axis=0)                            
        I2=np.delete(I2, np.s_[(A):(A+2*dt)], axis=1) 

else:
        root = tk.Tk()
        filez = np.array(fd.askopenfilenames(parent=root,title='Choose a file'))
        I = []
        for i in filez:
            I.append(cv2.imread(i,0))
            
a = np.shape(I)            
for i in range (0,4-1,1):
    I1 = I[i]
    I2 = I[i+1]
            

    xmax = I1.shape[0]
    ymax = I1.shape[1]
    
    #Window size
    print("Introduzca el tamaño de las ventanas de interrogación")
    w_width =np.int8(input())
    w_height = w_width
    
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
    print("Indique el estimador sub píxel que desee utilizar")
    print("1. Ajuste de Gauss con tres puntos")
    print("2. Ajuste parabólico")
    sbpx_method =np.int8(input())
    G = np.zeros([x_count,y_count])
    for i in range (0,x_count,1):
        for j in range (0,y_count,1):
            max_correlation = 0
            test_xmin = np.int32(xgrid[i]-w_width/2.)
            test_xmax = np.int32(xgrid[i]+w_width/2.)
            test_ymin = np.int32(ygrid[j]-w_height/2.)
            test_ymax = np.int32(ygrid[j]+w_height/2.)
            x_disp = 0
            y_disp = 0 
            test_ima = I1[test_xmin:test_xmax,test_ymin:test_ymax]
            test_imb = I2[(test_xmin-x_disp_max):(test_xmax+x_disp_max),(test_ymin-y_disp_max):(test_ymax+y_disp_max)]
            if np.size(test_imb)>0:
                c = normxcorr2(test_ima,test_imb)
                xpeak,ypeak = np.unravel_index(c.argmax(),c.shape)
            else:
               c = np.zeros([w_width+2*x_disp_max,w_height+2*y_disp_max])
               xpeak,ypeak = np.unravel_index(c.argmax(),c.shape)
             
            #Utilizamos estimadores sub píxel para mayor precisión
            if sbpx_method<2:
                f0_x = np.log(c[xpeak,ypeak])
                f1_x = np.log(c[xpeak-1,ypeak])
                f2_x = np.log(c[xpeak+1,ypeak])
                f0_y = np.log(c[xpeak,ypeak])
                f1_y = np.log(c[xpeak,ypeak-1])
                f2_y = np.log(c[xpeak,ypeak+1])
                xpeak_sbpx = xpeak + (f1_x-f2_x)/(2*f1_x+2*f2_x-4*f0_x)
                ypeak_sbpx = ypeak + (f1_y-f2_y)/(2*f1_y+2*f2_y-4*f0_x)
            else:
                xpeak_sbpx = xpeak
                ypeak_sbpx = ypeak
            
            xpeak1 = test_xmin + xpeak_sbpx - w_width/2. - x_disp_max
            ypeak1 = test_ymin + ypeak_sbpx - w_height/2. - y_disp_max
            dpx[i,j] = xpeak1 - xgrid[i]
            dpy[i,j] = ypeak1 - ygrid[j]
            G[i,j] = math.sqrt((dpx[i,j])**2+(dpy[i,j])**2)
    
    dpx = np.delete(dpx, np.s_[0], axis = 0)
    dpx = np.delete(dpx, np.s_[0], axis = 1)
    dpy = np.delete(dpy, np.s_[0], axis = 0)
    dpy = np.delete(dpy, np.s_[0], axis = 1)
    
    dpx = np.nan_to_num(dpx)
    dpy = np.nan_to_num(dpy)
    G = np.nan_to_num(G)
    
    [dpx2,dpy2,G] = st_dev_check(dpx,dpy,G,5,5,5)
    
    dpx2 = np.float32(dpx)
    dpy2 = np.float32(dpy)
    G2 = np.float32(G)
    
    dpx_smooth = cv2.medianBlur(dpx2,3)
    dpy_smooth = cv2.medianBlur(dpy2,3)
    G_smooth = cv2.medianBlur(G2,3)
    
    
    
    theta = np.zeros([dpx.shape[0],dpx.shape[1]])
    for i in range (0,dpx.shape[0]):
        for j in range(0,dpx.shape[1]):
            theta[i,j] = np.arctan(-dpx_smooth[i,j]/dpy_smooth[i,j])*180/np.pi
    
    #Dibuja cosas
    
    plt.figure()
    plt.quiver(dpy_smooth,-dpx_smooth, color = 'Green')
    plt.title("Campo de velocidades")
    
    plt.figure()
    plt.plot(dpy_smooth,-dpx_smooth,'o', color = 'black')
    plt.xlabel("Desplazamientos en X (píxel/frame)")
    plt.ylabel("Desplazamientos en Y (píxel/frame)")
    
    plt.figure()
    plt.imshow(dpy_smooth, interpolation = 'bilinear')
    plt.colorbar()
    plt.title("Desplazamientos según el eje X (píxel/frame)")
    
    plt.figure()
    plt.imshow(-dpx_smooth, interpolation = 'bilinear')
    plt.colorbar()
    plt.title("Desplazamientos según el eje y")
    
    plt.figure()
    plt.imshow(G_smooth,interpolation = 'bilinear')
    plt.colorbar()
    plt.title("Módulo de la velocidad")
    
    plt.figure()
    plt.imshow(theta, interpolation = 'bilinear')
    plt.title("Ángulo que forma el vector con la vertical")
    plt.colorbar()
    
    plt.figure()
    plt.hist(dpx_smooth)
    plt.xlabel("Valor del desplazamiento")
    plt.ylabel("Frecuencia")