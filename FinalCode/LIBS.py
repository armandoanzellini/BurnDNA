# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 20:35:54 2025

@author: Armando Anzellini

LIBS for Burn DNA with Mayra, Amy, Geri, and Anneke
"""

import pandas as pd
import matplotlib.pyplot as plt

direct = r"C:\Users\arman\Dropbox\Projects\DNA Burned Remains\Data\LIBS 2"
file   = "\\exported_therm_calcEA1.csv"

# LIBS data from single calcaneus
df = pd.read_csv(direct + file, index_col = 0, header=None)

# normalize all data to the largest peak in each
df = df/df.max()

# plot only the Nitrogen section
for spot in df.columns:
    fig, ax = plt.subplots()
    ax.plot(df[spot].loc[700:800], color='k', linewidth = .75) # plot
    ax.set_title(f'Spot {spot}') # title refering to point along bone
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (a.u.)')
    
    # add vlines at expected values
    ax.vlines(740, 0, .5)
    ax.vlines(750, 0, .5)
    # ax.vlines(746.8, 0, 100)
    
    ax.autoscale(enable=True, axis='x', tight=True) # set tight axis on X
    plt.show()
    
#
