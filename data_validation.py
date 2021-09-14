import numpy as np

def st_dev_check(u,v,G,n_x,n_y,n_g):
    mean_u = u.mean()
    mean_v = v.mean()
    mean_g = G.mean()
    st_dev_u = u.std()
    st_dev_v = v.std()
    st_dev_g = G.std()
    
    u_min = mean_u - n_x*st_dev_u
    u_max = mean_u + n_x*st_dev_u
    v_min = mean_v - n_y*st_dev_v
    v_max = mean_v + n_y*st_dev_v
    g_min = mean_g - n_g*st_dev_g
    g_max = mean_g + n_g*st_dev_g
    
    u[u<u_min] = 0
    u[u>u_max] = 0
    v[v<v_min] = 0
    v[v>v_max] = 0
    G[G<g_min] = 0
    G[G>g_max] = 0
    
    return u,v,G 