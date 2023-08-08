import numpy as np
import math

def st_dev_check(u, v, G, n_x, n_y, n_g):
    mean_u, mean_v, mean_g = np.mean(u), np.mean(v), np.mean(G)
    st_dev_u, st_dev_v, st_dev_g = np.std(u), np.std(v), np.std(G)

    u_min, u_max = mean_u - n_x * st_dev_u, mean_u + n_x * st_dev_u
    v_min, v_max = mean_v - n_y * st_dev_v, mean_v + n_y * st_dev_v
    g_min, g_max = mean_g - n_g * st_dev_g, mean_g + n_g * st_dev_g

    u = np.where((u >= u_min) & (u <= u_max), u, np.nan)
    v = np.where((v >= v_min) & (v <= v_max), v, np.nan)
    G = np.where((G >= g_min) & (G <= g_max), G, np.nan)

    return u, v, G
