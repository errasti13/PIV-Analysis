import imageio
import tkinter as tk
from tkinter import filedialog as fd
import numpy as np

root = tk.Tk()
filenames = np.array(fd.askopenfilenames(parent=root,title='Choose a file'))
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave('Imagenes/Karman/Mapa Vectorial/Mapa_Vectorial.gif', images)