# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:36:49 2025

@author: Armando Anzellini

Radial Hann Window
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

image = np.full([300,300], 255)

def radial_hann(image):
    Ly, Lx = image.shape
    
    windowed = np.zeros([Ly, Lx])

    pref = ((3*np.pi)/8 - 2/np.pi)**(-1/2)
    
    thresh = (min(Lx, Ly)/2)**2

    for i in range(Ly):
        Y = float(i) - (Ly/2)
        for k in range(Lx):
            X = float(k) - (Lx/2)
            
            # print(i,k)
            
            if (X**2 + Y**2) < thresh:
                numerator   = 2*np.pi*np.sqrt(X**2 + Y**2)
                denominator = min(Lx, Ly)
                values = pref * (1 + np.cos(numerator/denominator))
            else:
                values = 0
                
        windowed[k,i] = values
                     
        return windowed
    
plt.imshow(win := radial_hann(image), cmap='viridis')