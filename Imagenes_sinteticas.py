import numpy as np

def flujo_poiseuille(A,n,H,dt, I0):
    sigma=1/8.*(dt**2)
    y = np.linspace(0, H, A+2*dt)
    u = -1/(2*0.0000174)*(y**2-H*y)*2.6458 #Cambio de m/s a pixeles/s
    d = u*0.001
    d_max = np.int64(np.round(np.max(d)+1))
    d_round = np.int64(np.round(d))
        
    I1=np.zeros([A+2*dt,A+2*dt])
    I2=np.zeros([A+d_max+2*dt,A+2*dt])
        
    for i in range (0,n,1):
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
                 
    return I1, I2

def flujo_couette(A,n,H,dt,I0, U):
    sigma=1/8.*(dt**2)
    y = np.linspace(0, H, A+2*dt)
    u = (U/H)*y*2.6458 #Cambio de m/s a pixeles/s
    d = u*0.1
    d_max = np.int64(np.round(np.max(d)+1))
    d_round = np.int64(np.round(d))

    I1=np.zeros([A+2*dt,A+2*dt])
    I2=np.zeros([A+d_max+2*dt,A+2*dt])
        
    for i in range (0,n,1):
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
    
    return I1,I2

def desplazamiento_horizontal(A,n,dt,I0):
    sigma=1/8.*(dt**2)
    print("Número de píxeles que desee desplazar")
    d = np.int32(input())
    
    I1=np.zeros([A+2*dt,A+2*dt])
    I2=np.zeros([A+d+2*dt,A+2*dt])

    for i in range (0,n,1):
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
    
    return I1,I2