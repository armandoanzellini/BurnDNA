# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 21:02:11 2025

@author: Armando Anzellini

* Uses pyspectra which must be installed through pip (https://pypi.org/project/pyspectra/)

Need to actually use pyfasma-spc!!!
"""
import os
import sys
import glob
import spc
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# from pyspectra.readers.read_spc import read_spc

sys.path.append(r"C:\Users\arman\OneDrive\Documents\Python Scripts\SpectraUtils")

import SpecCalcMod as scm

# define directories to be used
direct_stafs = "C:\\Users\\arman\\Dropbox\\Projects\\DNA Burned Remains\\Data\\SHSU FTIR Data\\2025\\STAFS samples"
direct_utk   = "C:\\Users\\arman\\Dropbox\\Projects\\DNA Burned Remains\\Data\\SHSU FTIR Data\\2025\\UTK samples"
dna_directory  = "C:\\Users\\arman\\Dropbox\\Projects\\DNA Burned Remains\\Data\\"
outdir = r"C:\Users\arman\OneDrive\Documents\AuthorPapers (in progress)\Burn DNA\Submission\Figures"

# global variables
dpi=800     # dpi for figures

# figure out how many subdirectories there are since each sample has a subdirectory
subdirs_stafs = os.listdir(direct_stafs)
subdirs_utk   = os.listdir(direct_utk)

# get DNA file
dna_file = 'Pre-screening workbook.xlsx'

# get DNA data
cols = ['ID', 'Collection', 'Sample', 'Color', 'Qty','Allele']

dna = pd.read_excel(dna_directory + dna_file, sheet_name='ForCorr', skiprows=1, names=cols, index_col='ID')

# dna = dna.iloc[5:] # only use UTK for now until FTIR protocol is clear

# select only relevant columns from DNA data
dna = dna[['Color', 'Qty', 'Allele']].fillna(0)

# define necessary functions to read all spc in each subdir and select the best spectrum


def read_spc_dir(directory):
    """
    Read all spc files in a directory and put into a single Pandas DataFrame.
    All spc must be from same machine since indices of wavelength must match

    Parameters
    ----------
    directory : Path
        does not check if it's a real path, must do yourself.

    Returns
    -------
    df_spc : Pandas DataFrame
        filenames without extensions are the column names. Returns none if no
        SPC files found in that directory

    """

    # get list of files in directory that have an .spc extension
    if directory.endswith("\\"):
        file_paths = glob.glob(directory + '*.spc')
    elif not directory.endswith("\\"):
        file_paths = glob.glob(directory + '\\*.spc')
        
    # check if any SPC files are found and exit if none
    if len(file_paths) == 0:
        print(f"No SPC Files found in the given directory: {directory}")
        return pd.DataFrame()

    # pop out the first file and separate into a name and a list
    file1 = file_paths[0]
    files = file_paths[1:]
    
    # get name of the first file separately without extension for column name
    f1name = os.path.splitext(os.path.basename(file1))[0] 
    
    # open first file as a pd dataframe with filename as column
    read_file = spc.File(file1)
    df_spc = pd.DataFrame(list(zip(read_file.x, read_file.sub[0].y)),
                columns=[read_file.xlabel, f1name]).set_index(read_file.xlabel)
    
    # now loop through the rest of the files in the directory
    for file in files:
        # get name for column
        fname = os.path.splitext(os.path.basename(file))[0]
        
        # open file as Pandas series while assigning to column in df
        read_file = spc.File(file)
        
        temp_df = pd.DataFrame(list(zip(read_file.x, read_file.sub[0].y)),
                    columns=[read_file.xlabel, fname]).set_index(read_file.xlabel)
        
        df_spc = df_spc.join(temp_df) #join to make sure it joins on index (wavenumeber)
         
    # now return new Pandas DataFrame with all files read
    return df_spc
    
def best_ATRFTIR_spec(directory):
    
    samp = read_spc_dir(directory)
    
    # check that read_spc_dir returns actual df
    if samp.empty:
       return samp
    
    # select the spectrum with the largest absorbtion peak as it is the most
    # accurate (DOI: 10.1177/1744259111423367)
    # find the largest maximum around 1000 since thats where we expect it to be
    # any other peak larger than that one means the signal is bad
    best_rep = samp[samp.index < 1200].max().idxmax()
    
    samp = samp[best_rep] # now make sure you have only selected the best spectrum
    
    # return series with only the selected spectrum and name from column name
    return samp

def avg_ATRFTIR_spec(directory):
    samp = read_spc_dir(directory)
    
    # check that read_spc_dir returns actual df
    if samp.empty:
       return samp
    
    # average the spectra together
    samp = samp.mean(axis=1)
    
    # return series with only the selected spectrum and name from column name
    return samp

def average_spectra(base, plot):
    # open STAFS FTIR spectra
    subdir1_stafs = subdirs_stafs[0]
    subdir_stafs = subdirs_stafs[1:]
    
    # Read first spc to make DF and select best, convert to DF with name of folder
    df_spc = avg_ATRFTIR_spec(direct_stafs + "\\" + subdir1_stafs).to_frame('SHSU-0')
    
    # loop rest of the subdirs to append the best spectra per sample to the df
    for folder in subdir_stafs:
        if avg_ATRFTIR_spec(direct_stafs + "\\" + folder).empty:
            pass
        else:
            col_name = folder.replace('Sample ', 'SHSU-')
            df_spc[col_name] = avg_ATRFTIR_spec(direct_stafs + "\\" + folder)
            
    # now open UTK FTIR spectra
    # loop thorugh subdirs-utk to append the best spectra per sample to the df
    for folder in subdirs_utk:
        if avg_ATRFTIR_spec(direct_utk + "\\" + folder).empty:
            pass
        else:
            col_name = folder.replace('Sample ', 'UTK-')
            df_spc[col_name] = avg_ATRFTIR_spec(direct_utk + "\\" + folder)
        
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
    
    for column in df_spc.columns[:-1]: # skip the last column which is Wavelength
    
        amIPO4area = scm.amIPO4area(df_spc, 'Wavelength', column, smooth=True,  plot=plot, savefigdir=None) #r'C:\Users\arman\Desktop\Results\avg')
        cp         = scm.cp_ratio(df_spc, 'Wavelength', column, vers='ftir', base=base, smooth=True, plot=plot, savefigdir=None) #r'C:\Users\arman\Desktop\Results\avg')
        amIPO4     = scm.amIPO4(df_spc, 'Wavelength', column, vers='ftir', base=base, smooth=True,  plot=plot, savefigdir=None) # r'C:\Users\arman\Desktop\Results\avg')
        
        # add values to the dictionary
        ind_dict[column] = [amIPO4area, cp, amIPO4]
    
    # add indices to the DNA df
    ind_df = pd.DataFrame.from_dict(ind_dict, orient='index', columns=['AmIPO4area', 'CP', 'AmIPO4'])
    
    df = dna.join(ind_df) # join on index
    
    # now plot each index to the dna values
    # for col in ['AmIPO4area', 'CP', 'AmIPO4']:
    #     plt.scatter(df[col],df['Qty'], c='k')
    #     plt.title(f'Qty v. {col}')
    #     plt.show()
        
    # for col in ['AmIPO4area', 'CP', 'AmIPO4']:
    #     plt.scatter(df[col],df['Allele'], c='k')
    #     plt.title(f'Allele v. {col}')
    #     plt.show()
    
    # potentially a cut-off for both area and peak amIpo4 at .165

    for col in ['AmIPO4area', 'AmIPO4', 'CP']:
        fig, ax = plt.subplots()
        # plot
        ax.scatter(df[col],df['Allele'], c='k', alpha=0.5)
        
        # title and axes
        if 'area' in col:
            label = 'Amide I/PO$_{4}$ Ratio by Area'
        elif 'CP' in col:
            label = 'CO$_{3}$/PO$_{4}$'
        else:
            label = 'Amide I/PO$_{4}$ Ratio by Peak Heights'
            
        
        
        # show threshold
        if 'CP' in col:
            thresh = 0.330
        else:
            thresh = 0.165
        
        ax.vlines(thresh, 0, df['Allele'].max(), linestyles='--', colors='k')
        
        ax.set_title('ATR-FTIR Threshold Prediction', family='serif')
        ax.set_ylabel('STR Recovery %', family='serif')
        ax.set_xlabel(label, family='serif')
        
        # prediction annotation
        correct   = len(df[(df['Allele'] > 60) & (df[col] > thresh)])
        false_neg = len(df[(df['Allele'] > 60) & (df[col] <= thresh)])
        total = len(df[df['Allele'] > 60])
        predict   = correct/total * 100
        wrong     = false_neg/total * 100
        
        ax.annotate("Prediction ~ " + f"{predict: .1f}%", 
                    xy=(0.19, 40),  xycoords='data', family="serif")
        ax.annotate("False Negative ~ " + f"{wrong: .1f}%", 
                    xy=(0.19, 33.5),  xycoords='data', family="serif")
        ax.annotate("Threshold ~ " + f"{thresh: .3f}", 
                    xy=(0.19, 26.5),  xycoords='data', family="serif")
        
        plt.savefig(outdir + f'\\Allele_{col}_avg.png', dpi=dpi, transparent=False)
        plt.show()
    
    # polyfit from numpy for AmIPO4 with Log(Qty)
    dat = df[df['Qty']>0] # [(df['Qty'] < 10) & (df['AmIPO4']<.35)].sort_values(by='AmIPO4area')
    
    for col in ['AmIPO4area', 'AmIPO4', 'CP']:
        X = dat[col].to_numpy()
        y = dat['Qty'].to_numpy()
        
        # p = np.polynomial.polynomial.polyfit(X, np.log(y), deg=1)
        
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(X, np.log(y))
        p = [intercept, slope]
        
        residuals = y - np.exp(p[1] * X + p[0])
        ci        = np.quantile(residuals, 0.9)
        line_x    = np.linspace(x_min := 0, x_max := X.max()+0.05)
        pred      = np.exp(p[1] * line_x + p[0])
        
        # find CI
        upper_ci  = np.exp(p[1] * line_x + p[0] + ci)
        lower_ci  = np.exp(p[1] * line_x + p[0] - ci)
        
        fig, ax = plt.subplots()
        ax.scatter(df[col], df['Qty'], c='k', alpha=0.5)
        ax.semilogy(line_x, pred, color='k', linestyle='--')
        ax.semilogy(line_x, upper_ci, color='k', linestyle=':')
        ax.semilogy(line_x, lower_ci, color='k', linestyle=':')
        ax.fill_between(line_x, upper_ci, lower_ci, alpha=0.25, color='gray')
        
        # set axes labels, extent, and title
        if 'area' in col:
            label = 'Amide I/PO$_{4}$ Ratio by Area'
        elif 'CP' in col:
            label = 'CO$_{3}$/PO$_{4}$'
        else:
            label = 'Amide I/PO$_{4}$ Ratio by Peak Heights'
            
        ax.set_title('ATR-FTIR Quantity Prediction', family='serif')
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel('Log(Qty)', family='serif')
        ax.set_xlabel(label, family='serif')
        
        # add function to plot
        ax.annotate(f"ln(y) = {p[1]: .3f}x + {p[0]: .3f}", 
                    xy=(0.245, 1*10e-8),  xycoords='data', family="serif")
        ax.annotate("R$^{2}$ = " + str(round(r_value**2, 2)),
                    xy=(0.245, 8*10e-8), xycoords='data', family="serif")
        ax.annotate("SEE = " + str(round(std_err, 2)),
                    xy=(0.245, 1*10e-9), xycoords='data', family="serif")
        
        # set extent
        ax.set_ylim(bottom = 8*10e-11)
        
        plt.savefig(outdir + f'\\Qty_{col}_avg.png', dpi=dpi, transparent=False)
        # plt.show()
        

    return df, df_spc

def best_spectrum(base, plot):
    # open STAFS FTIR spectra
    subdir1_stafs = subdirs_stafs[0]
    subdir_stafs = subdirs_stafs[1:]
    
    # Read first spc to make DF and select best, convert to DF with name of folder
    df_spc = best_ATRFTIR_spec(direct_stafs + "\\" + subdir1_stafs).to_frame('SHSU-0')
    
    # loop rest of the subdirs to append the best spectra per sample to the df
    for folder in subdir_stafs:
        if best_ATRFTIR_spec(direct_stafs + "\\" + folder).empty:
            pass
        else:
            col_name = folder.replace('Sample ', 'SHSU-')
            df_spc[col_name] = best_ATRFTIR_spec(direct_stafs + "\\" + folder)
            
    # now open UTK FTIR spectra
    # loop thorugh subdirs-utk to append the best spectra per sample to the df
    for folder in subdirs_utk:
        if best_ATRFTIR_spec(direct_utk + "\\" + folder).empty:
            pass
        else:
            col_name = folder.replace('Sample ', 'UTK-')
            df_spc[col_name] = best_ATRFTIR_spec(direct_utk + "\\" + folder)
        
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
    
    for column in df_spc.columns[:-1]: # skip the last column which is Wavelength
    
        amIPO4area = scm.amIPO4area(df_spc, 'Wavelength', column, smooth=True, plot=plot, savefigdir=None) #r'C:\Users\arman\Desktop\Results\best')
        cp         = scm.cp_ratio(df_spc, 'Wavelength', column, vers='ftir', base=base, smooth=False, plot=plot, savefigdir=None) #r'C:\Users\arman\Desktop\Results\best')
        amIPO4     = scm.amIPO4(df_spc, 'Wavelength', column, vers='ftir', base=base, smooth=True, plot=plot, savefigdir=None) #r'C:\Users\arman\Desktop\Results\best')
        
        # add values to the dictionary
        ind_dict[column] = [amIPO4area, cp, amIPO4]
    
    # add indices to the DNA df
    ind_df = pd.DataFrame.from_dict(ind_dict, orient='index', columns=['AmIPO4area', 'CP', 'AmIPO4'])
    
    df = dna.join(ind_df) # join on index
    
    # now plot each index to the dna values
    # for col in ['AmIPO4area', 'CP', 'AmIPO4']:
    #     plt.scatter(df[col],df['Qty'], c='k')
    #     plt.title(f'Qty v. {col}')
    #     plt.show()
        
    # for col in ['AmIPO4area', 'CP', 'AmIPO4']:
    #     plt.scatter(df[col],df['Allele'], c='k')
    #     plt.title(f'Allele v. {col}')
    #     plt.show()
    
    # potentially a cut-off for both area and peak amIpo4 at .165
    # try .17 for both
    for col in ['AmIPO4area', 'AmIPO4', 'CP']:
        fig, ax = plt.subplots()
        # plot
        ax.scatter(df[col],df['Allele'], c='k', alpha=0.5)
        
        # title and axes
        if 'area' in col:
            label = 'Amide I/PO$_{4}$ Ratio by Area'
        elif 'CP' in col:
            label = 'CO$_{3}$/PO$_{4}$'
        else:
            label = 'Amide I/PO$_{4}$ Ratio by Peak Heights'
            
        
        
        # show threshold
        if 'CP' in col:
            thresh = 0.330
        else:
            thresh = 0.165
        
        ax.vlines(thresh, 0, df['Allele'].max(), linestyles='--', colors='k')
        
        ax.set_title('ATR-FTIR Prediction', family='serif')
        ax.set_ylabel('Allele % Match', family='serif')
        ax.set_xlabel(label, family='serif')
        
        # prediction annotation
        correct   = len(df[(df['Allele'] > 60) & (df[col] > thresh)])
        false_neg = len(df[(df['Allele'] > 60) & (df[col] <= thresh)])
        total = len(df[df['Allele'] > 60])
        predict   = correct/total * 100
        wrong     = false_neg/total * 100
        
        ax.annotate("Prediction ~ " + f"{predict: .1f}%", 
                    xy=(0.2, 40),  xycoords='data', family="serif")
        ax.annotate("False Negative ~ " + f"{wrong: .1f}%", 
                    xy=(0.2, 33.5),  xycoords='data', family="serif")
        ax.annotate("Threshold ~ " + f"{thresh: .3f}", 
                    xy=(0.2, 26.5),  xycoords='data', family="serif")
        
        plt.savefig(outdir + f'\\Allele_{col}_avg.png', dpi=dpi, transparent=False)
        plt.show()
    
    # polyfit from numpy for AmIPO4 with Log(Qty)
    dat = df[df['Qty']>0] # [(df['Qty'] < 10) & (df['AmIPO4']<.35)].sort_values(by='AmIPO4area')
    
    for col in ['AmIPO4area', 'AmIPO4', 'CP']:
        X = dat[col].to_numpy()
        y = dat['Qty'].to_numpy()
        
        # p = np.polynomial.polynomial.polyfit(X, np.log(y), deg=1)
        
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(X, np.log(y))
        p = [intercept, slope]
        
        residuals = y - np.exp(p[1] * X + p[0])
        ci        = np.quantile(residuals, 0.9)
        line_x    = np.linspace(x_min := 0, x_max := X.max()+0.05)
        pred      = np.exp(p[1] * line_x + p[0])
        
        # find CI
        upper_ci  = np.exp(p[1] * line_x + p[0] + ci)
        lower_ci  = np.exp(p[1] * line_x + p[0] - ci)
        
        fig, ax = plt.subplots()
        ax.scatter(df[col], df['Qty'], c='k', alpha=0.5)
        ax.semilogy(line_x, pred, color='k', linestyle='--')
        ax.semilogy(line_x, upper_ci, color='k', linestyle=':')
        ax.semilogy(line_x, lower_ci, color='k', linestyle=':')
        ax.fill_between(line_x, upper_ci, lower_ci, alpha=0.25, color='gray')
        
        # set axes labels, extent, and title
        if 'area' in col:
            label = 'Amide I/PO$_{4}$ Ratio by Area'
        elif 'CP' in col:
            label = 'CO$_{3}$/PO$_{4}$'
        else:
            label = 'Amide I/PO$_{4}$ Ratio by Peak Heights'
            
        ax.set_title('ATR-FTIR Quantity Prediction', family='serif')
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel('Log(Qty)', family='serif')
        ax.set_xlabel(label, family='serif')
        
        # add function to plot
        ax.annotate(f"ln(y) = {p[1]: .3f}x + {p[0]: .3f}", 
                    xy=(0.245, 1*10e-8),  xycoords='data', family="serif")
        ax.annotate("R$^{2}$ = " + str(round(r_value**2, 2)),
                    xy=(0.245, 8*10e-8), xycoords='data', family="serif")
        ax.annotate("SEE = " + str(round(std_err, 2)),
                    xy=(0.245, 1*10e-9), xycoords='data', family="serif")
        
        # set extent
        ax.set_ylim(bottom = 8*10e-11)
        
        # plt.savefig(r'C:\Users\arman\Desktop\Results' + f'\\Qty_{col}_best.png', dpi=300, transparent=False)
        plt.show()
        
    return df, df_spc

avgdf, df_aspc = average_spectra(base = False, plot = False)

# run log-reg with dummy variables for allele (>60, <60) and qty (yes, no)
avgdf['Allele_Cat'] = [0 if a < 60. else 1 for a in avgdf['Allele']]
avgdf['Qty_Cat']    = [0 if a == 0.  else 1 for a in avgdf['Qty']] 

# run logit for avg
for col in ['AmIPO4area', 'CP']:
    for cat in ['Allele_Cat', 'Qty_Cat']:
        X = sm.add_constant(avgdf[col]) # add a constant for intercept of logit
        y = avgdf[cat]
        
        model = sm.Logit(y, X)
        
        result = model.fit(method='newton')
        
        conf_matrix = result.pred_table()
        
        print(result.summary2())
        
        print(conf_matrix)
        
        TP = conf_matrix[0][0]
        TN = conf_matrix[1][1]
        FN = conf_matrix[0][1]
        FP = conf_matrix[1][0]
        
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        
        # create x for plotting
        line_x    = np.linspace(x_min := 0, x_max := X[col].max())
        
        # plot
        fig, ax = plt.subplots()
        
        # plot binarized data and logistic function from prediction
        ax.scatter(avgdf[col], avgdf[cat], c='k', alpha=0.5)
        ax.plot(line_x, result.predict(sm.add_constant(line_x)), 'k', linestyle = '--')
        
        # set labels and title
        if cat == 'Allele_Cat':
            y_label = 'STR Recovery'
        else:
            y_label = 'DNA Extraction Quantity'
            
        if 'area' in col:
            x_label = 'Amide I/PO$_{4}$ Ratio by Area'
        elif 'CP' in col:
            x_label = 'CO$_{3}$/PO$_{4}$'
        else:
            x_label = 'Amide I/PO$_{4}$ Ratio by Peak Heights'
            
        ax.set_title('ATR-FTIR Logistic Regression on DNA', family='serif')
        ax.set_ylabel(y_label, family='serif')
        ax.set_xlabel(x_label, family='serif')
        
        # add annotation of log_odds
        if 'CP' in col:
            x_ann = 0.27
        else:
            x_ann = 0.20
        
        func = f"{result.params['const']: .2f} + {result.params[col]: .2f}x"
        
        ax.annotate(f'Odds = e$^{{{func}}}$',
                    xy=(x_ann, .2),  xycoords='data', family="serif", fontsize=12)
        ax.annotate(f'Sensitivity = {Sensitivity: .2f}',
                    xy=(x_ann, .12),  xycoords='data', family="serif")
        ax.annotate(f'Specificity = {Specificity: .2f}',
                    xy=(x_ann, .04),  xycoords='data', family="serif")

        plt.savefig(outdir + f'\\{cat}_{col}_Logistic_avg.png', dpi=dpi, transparent=False)
        plt.show()

# run log-reg with dummy variables for allele (>60, <60) and qty (yes, no)
avgdf['Allele_Cat'] = [0 if a < 60. else 1 for a in avgdf['Allele']]
avgdf['Qty_Cat']    = [0 if a == 0.  else 1 for a in avgdf['Qty']] 

# run logit for best
for col in ['AmIPO4area', 'AmIPO4', 'CP']:
    for cat in ['Allele_Cat', 'Qty_Cat']:
        X = sm.add_constant(avgdf[col]) # add a constant for intercept of logit
        y = avgdf[cat]
        
        model = sm.Logit(y, X)
        
        result = model.fit(method='newton')
        
        conf_matrix = result.pred_table()
        
        print(result.summary2())
        
        print(conf_matrix)
        
        TP = conf_matrix[0][0]
        TN = conf_matrix[1][1]
        FN = conf_matrix[0][1]
        FP = conf_matrix[1][0]
        
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        
        # create x for plotting
        line_x    = np.linspace(x_min := 0, x_max := X[col].max())
        
        # plot
        fig, ax = plt.subplots()
        
        # plot binarized data and logistic function from prediction
        ax.scatter(avgdf[col], avgdf[cat], c='k', alpha=0.5)
        ax.plot(line_x, result.predict(sm.add_constant(line_x)), 'k', linestyle = '--')
        
        # set labels and title
        if cat == 'Allele_Cat':
            y_label = 'Allele Recovery'
        else:
            y_label = 'DNA Extraction Quantity'
            
        if 'area' in col:
            x_label = 'Amide I/PO$_{4}$ Ratio by Area'
        elif 'CP' in col:
            x_label = 'CO$_{3}$/PO$_{4}$'
        else:
            x_label = 'Amide I/PO$_{4}$ Ratio by Peak Heights'
            
        ax.set_title('ATR-FTIR Logistic Regression on DNA', family='serif')
        ax.set_ylabel(y_label, family='serif')
        ax.set_xlabel(x_label, family='serif')
        
        # add annotation of log_odds
        if 'CP' in col:
            x_ann = 0.27
        else:
            x_ann = 0.20
        
        func = f"{result.params['const']: .2f} + {result.params[col]: .2f}x"
        
        ax.annotate(f'Odds = e$^{{{func}}}$',
                    xy=(x_ann, .2),  xycoords='data', family="serif", fontsize=12)
        ax.annotate(f'Sensitivity = {Sensitivity: .2f}',
                    xy=(x_ann, .12),  xycoords='data', family="serif")
        ax.annotate(f'Specificity = {Specificity: .2f}',
                    xy=(x_ann, .04),  xycoords='data', family="serif")

        # plt.savefig(r'C:\Users\arman\Desktop\Results' + f'\\{cat}_{col}_Logistic_best.png', dpi=300, transparent=False)
        plt.show()
