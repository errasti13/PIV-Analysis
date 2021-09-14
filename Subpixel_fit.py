import numpy as np

def gaussian_fit(a,b,c):
    global xgauss, ygauss
    if a<c.shape[0]-1:
        if b<c.shape[1]-1:
            f0_x = np.log(c[a,b])
            f1_x = np.log(c[a-1,b])
            f2_x = np.log(c[a+1,b])
            f0_y = np.log(c[a,b])
            f1_y = np.log(c[a,b-1])
            f2_y = np.log(c[a,b+1])
            xgauss = a + (f1_x-f2_x)/(2*f1_x+2*f2_x-4*f0_x)
            ygauss = b + (f1_y-f2_y)/(2*f1_y+2*f2_y-4*f0_y)
    return xgauss,ygauss

def parabolic_fit(a,b,c):
    xpeak_parab = a + (c[a-1,b]-c[a+1,b])/(2*c[a-1,b]-4*c[a,b]+2*c[a+1,b])
    ypeak_parab = b + (c[a,b-1]-c[a,b+1])/(2*c[a,b-1]-4*c[a,b]+2*c[a,b+1])
    
    return xpeak_parab,ypeak_parab
