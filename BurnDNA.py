# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:05:24 2021

@author: Armando Anzellini
"""
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, 'C:\\Users\\aanzellini\\OneDrive\\Documents\\Python Scripts\\IsoSpec\\')
from SpecCalcMod import amIPO4

direct = "C:\\Users\\aanzellini\\OneDrive\\Documents\\AuthorPapers (in progress)\\Burn DNA\\Data\\CSV\\"
folder = "Powder"

os.chdir(direct + folder)

# Get list of files from directory and compile a list of only csv files
files = glob.glob('*.csv')

# Compile the data of each file into a pandas dataframe
data = []
for i in files:
    sampname = 'S' + ''.join([x for x in i if x.isdigit()])
    samp     = pd.read_csv(i ,names = ['wavelength', sampname], 
                               header=None,
                                skiprows=2,
                                  index_col=False)
         
    data.append(samp)

df = pd.concat(data, axis = 1)

df = df.loc[:,~df.columns.duplicated()]

# Define X and Y to plot all spectra
X = 'wavelength'
#Y = df.columns.to_list()[1:]

for Y in df.columns.to_list()[1:]:
    # Create subplot
    fig, ax = plt.subplots(figsize= [10, 4])
    
    # Spectral plot
    ax.plot(df[X], df[Y], '-', color = 'gray')
    
    # Fix axis direction, set title, add labels
    ax.invert_xaxis()
    ax.set_title(Y)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
    ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
    
    # Set minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.show()

'''
FIX S8
'''
apr_dict = {} # create dictionary of ratio values to add to DNA quant dataframe

for y_col in df.columns.to_list()[1:]:
    key = y_col
    val = amIPO4(df, 'wavelength' , y_col)
    apr_dict[key] = val    
    
quants = pd.read_csv(direct + 'QuantResults.csv')

quants = quants.set_index('Sample')

# append apr_dict to quant dataframe for furhter analysis
quants['AmIPO4'] = pd.Series(apr_dict)

# append quants to end of df by Snam and by Snum and make those the wavelength row
human = 'Total Human DNA Qty'

quant_data = quants[human]

df_t = df.set_index('wavelength').T

df = df_t.join(quant_data)

# Do Partial Least Squares Regression on the signals (Snam)
pls = PLSRegression(n_components = 1)
pls.fit(df.iloc[:, :-1], df[human])

plt.scatter(pls.transform(df.iloc[:, :-1]), df[human], c=df[human]/15)
#plt.yscale('log')
plt.show()

# PLS doesn't seem to work too well
# Trying with improved AmIPO4
plt.scatter(quants['AmIPO4'], quants[human])
#plt.yscale('log')
plt.show()


# Now try with calculating %N


