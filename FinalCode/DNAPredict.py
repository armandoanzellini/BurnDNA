# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 08:40:55 2025

@author: Armando Anzellini

This one is to run the FTIR DNA prediction models
"""

import os
import re
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pyspectra.readers.read_spc import read_spc

sys.path.append(r"C:\Users\arman\OneDrive\Documents\Python Scripts\BurnDNA")

import SpecCalcMod as scm

#-----------------------------------------------------------------------------
# Utility and Read Functions
#-----------------------------------------------------------------------------
def read_spc_dir(file_paths):
    """
    Read all spc files in a directory and put into a single Pandas DataFrame.
    All spc must be from same machine since indices of wavelength must match

    Parameters
    ----------
    file_paths : list
        list of full file paths including the SPC extension

    Returns
    -------
    df_spc : Pandas DataFrame
        filenames without extensions are the column names. Returns none if no
        SPC files found in that directory

    """

    # pop out the first file and separate into a name and a list
    file1 = file_paths[0]
    files = file_paths[1:]
    
    # get name of the first file separately without extension for column name
    f1name = os.path.splitext(os.path.basename(file1))[0] 
    
    # open first file as a pd dataframe with filename as column
    df_spc = pd.DataFrame(read_spc(file1), columns = [f1name])
    
    # now loop through the rest of the files in the directory
    for file in files:
        # get name for column
        fname = os.path.splitext(os.path.basename(file))[0]
        
        # open file as Pandas series while assigning to column in df
        df_spc[fname] = read_spc(file)
         
    # now return new Pandas DataFrame with all files read
    return df_spc

def best_ATRFTIR_spec(file_paths):
    """
    Select the spectrum with the largest absorbtion peak at approximately 
    1000 cm-1 as it is the most accurate (DOI: 10.1177/1744259111423367)
    Any other peak larger than that one means the signal is bad.

    Parameters
    ----------
    file_paths : list
        list of full file paths including the SPC extension

    Returns
    -------
    samp : Pandas DataFrame
        DataFrame of the best spectrum as definied by the peak height at 
        approx 1000 cm-1

    """
    
    samp = read_spc_dir(file_paths)
    
    # check that read_spc_dir returns actual df
    if samp.empty:
       return samp
    
    # find the largest maximum around 1000 since thats where we expect it to be
    best_rep = samp[samp.index < 1200].max().idxmax()
    
    samp = samp[best_rep] # now make sure you have only selected the best spectrum
    
    # return series with only the selected spectrum and name from column name
    return samp

def best_spectrum(dictionary, base, plot, outdir=None):
    """
    

    Parameters
    ----------
    dictionary : TYPE
        DESCRIPTION.
    base : TYPE
        DESCRIPTION.
    plot : TYPE
        DESCRIPTION.

    Returns
    -------
    ind_df : TYPE
        DESCRIPTION.
    df_spc : TYPE
        DESCRIPTION.

    """
    # sample names
    s_names = list(dictionary.keys())
    
    # Read first spc to make DF and select best, convert to DF with name of folder
    df_spc = best_ATRFTIR_spec(dictionary[s_names[0]]).to_frame(s_names[0])
    
    # loop rest of the subdirs to append the best spectra per sample to the df
    for name in s_names[1:]:
        if best_ATRFTIR_spec(dictionary[name]).empty:
            pass
        else:
            col_name = name
            df_spc[col_name] = best_ATRFTIR_spec(dictionary[name])
        
    # select only utk for now since ftir protocol was unsure for stafs
    # df_spc = df_spc.iloc[:,6:]
    
    # df_spc.plot(legend=False)
    # plt.show()
    
    
    # for scm dataframe is assumed to have a wavelength col rather than index
    df_spc['Wavelength'] = df_spc.index
    
    # =============================================================================
    # Baseline correct the all the spectra and plot
    # def base(df, x_col, y_col, cutoff=None):
    #     if cutoff:
    #         dat = df[df[x_col] < cutoff][y_col]
    #     elif not cutoff:
    #         dat = df[y_col]
        
    #     # fit a polyline to that section
    #     base = peakutils.baseline(dat) # need to turn into a series for peakutils to work
        
    #     # subtract baseline from data
    #     dat -= base
        
    #     # update df using pandas df update method
    #     df.update(dat)
        
    #     #df.plot(x_col, y_col)
        
    #     return df
    
    
    # # apply baseline correction to the df_spc dataframe
    # for column in df_spc.columns[:-1]:
    #     base(df_spc, 'Wavelength', column, cutoff=None)
    # =============================================================================
        
    # normalize all spectra (not Wavelength) by largest peak value and get wavelength as a column again
    norm_df = df_spc.iloc[:,:-1]/df_spc.iloc[:,:-1].max()
    
    norm_df['Wavelength'] = norm_df.index
    
    # run all with smooth = True and then can try without smoothing
    ind_dict = {}
    
    # To test functions in SpecCalcMod
    # df = df_spc[['Reference', 'Wavelength']]
    # x_col = 'Wavelength'
    # y_col = 'Reference'
    # smooth = True
    # baseline = True
    # vers = 'ftir'
    
    # figures per sample go in <outdir>/best; None disables saving
    figdir = None
    if outdir is not None:
        figdir = Path(outdir) / "best"
        figdir.mkdir(parents=True, exist_ok=True)
        figdir = str(figdir)  # scm.amIPO4area concatenates strings

    for column in df_spc.columns[:-1]: # skip the last column which is Wavelength

        amIPO4area = scm.amIPO4area(df_spc, 'Wavelength', column, smooth=True, plot=plot, savefigdir=figdir)
        # cp         = scm.cp_ratio(df_spc, 'Wavelength', column, vers='ftir', base=base, smooth=True, plot=plot, savefigdir=None) #r'C:\Users\arman\Desktop\Results\best')
        # amIPO4     = scm.amIPO4(df_spc, 'Wavelength', column, vers='ftir', base=base, smooth=True, plot=plot, savefigdir=None) #r'C:\Users\arman\Desktop\Results\best')
        
        # add values to the dictionary
        ind_dict[column] = [amIPO4area] #, cp, amIPO4]
    
    # add indices to the DNA df
    ind_df = pd.DataFrame.from_dict(ind_dict, orient='index', columns=['AmIPO4area']) #, 'CP', 'AmIPO4'])
    
    # now plot each index to the dna values
    # for col in ['AmIPO4area', 'CP', 'AmIPO4']:
    #     plt.scatter(df[col],df['Qty'], c='k')
    #     plt.title(f'Qty v. {col}')
    #     plt.show()
        
    # for col in ['AmIPO4area', 'CP', 'AmIPO4']:
    #     plt.scatter(df[col],df['Allele'], c='k')
    #     plt.title(f'Allele v. {col}')
    #     plt.show()
        
    return ind_df, df_spc

#-----------------------------------------------------------------------------
# Models
#-----------------------------------------------------------------------------
def AmIPO4area_logit(value):
    exp = -4.45 + (30.13*value)
    odds = np.exp(exp)
    prob = odds/(1+odds)
    
    return prob

def AmIPO4area_thresh(value):
    thresh = 0.165
    
    if value >= thresh:
        return 'Match Possible'
    elif value < thresh:
        return 'No Match Possible'
    
def AmIPO4area_qty(value):
    predict = (26.627*value) - 7.995
    print(predict)
    return np.exp(predict)

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    # env.json (next to this file) holds the data directory path
    env_path = Path(__file__).resolve().parent / "env.json"
    with open(env_path, 'r') as f:
        direct = Path(json.load(f)["data_path"])

    # Output directory for figures and results in excel
    outdir = direct / "Results"
    outdir.mkdir(parents=True, exist_ok=True)

    # case-insensitive match since instrument files use .SPC
    file_paths = sorted(p for p in direct.iterdir() if p.suffix.lower() == '.spc')

    # group replicates by sample name; exact grouping so "Sample 1" does not
    # also pick up "Sample 10"
    samps = {}

    for file in file_paths:
        match = re.search(r'^(.+?)\srep\s\d+$', file.stem, flags=re.IGNORECASE)
        if match is None:
            print(f"Skipping unrecognized filename: {file.name}")
            continue
        samps.setdefault(match.group(1), []).append(str(file))

    indices, samps = best_spectrum(samps, base=False, plot=True, outdir=outdir)

    for samp in samps.iloc[:,:16]:
        fig, ax = plt.subplots()

        ax.plot(samps[samp], color='k', linewidth=0.5)
        ax.set_title(samp)
        ax.invert_xaxis()
        ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax.set_ylabel('Absorbance')

        fig.savefig(outdir / f"{samp}.png")
        plt.close(fig)

    
    # add values to indices df
    indices['Allele Logit Prob'] = indices['AmIPO4area'].apply(AmIPO4area_logit)
    indices['Allele Threshold']  = indices['AmIPO4area'].apply(AmIPO4area_thresh)  
    indices['Qty Estimate'] = indices['AmIPO4area'].apply(AmIPO4area_qty)
    
    # output results to a table
    with pd.ExcelWriter(outdir / 'Results.xlsx') as writer:
       indices.to_excel(writer, sheet_name='Sheet1')


