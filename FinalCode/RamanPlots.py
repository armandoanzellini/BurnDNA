# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:58:39 2026

@author: Armando Anzellini
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# read Raman files for both 1064 (CSV) and 785 (SPC) to make publication plot
folder_1064 = r"C:\Users\arman\Dropbox\Projects\DNA Burned Remains\Data\Armando Raman 1064"
outdir = r"C:\Users\arman\OneDrive\Documents\AuthorPapers (in progress)\Burn DNA\Submission\Figures"
# Get DNA data
# upload DNA data
dna_directory  = "C:\\Users\\arman\\Dropbox\\Projects\\DNA Burned Remains\\Data\\"

dna_file = 'Pre-screening workbook.xlsx'

# get DNA data
cols = ['ID', 'Collection', 'Sample', 'Color', 'Qty','Allele']

dna = pd.read_excel(dna_directory + dna_file, sheet_name='ForCorr', skiprows=6, names=cols, index_col='ID')

# select only relevant columns from DNA data
dna = dna[['Color', 'Qty', 'Allele']].fillna(0)

# Read the 1064 raman data
filepaths = glob.glob(folder_1064 + '\\*.csv')

# read the first file with the file title and number as column name
fnames = os.path.splitext(os.path.basename(filepaths[0]))[0]
fname = fnames.replace(',', '-')

# when reading index Raman Shift
df = pd.read_csv(filepaths[0],
                skiprows=98,
                index_col=['Raman Shift'],
                usecols=['Raman Shift',
                         'Dark', 
                         'Raw data #1',
                         'Dark Subtracted #1']
                ).rename(columns={'Dark Subtracted #1' : fname})

# Calculate uncorrected spectrum to show fluorescence
df_raw = pd.DataFrame({fname +'_raw' : df['Raw data #1'] - df['Dark']})

# drop unnecessary columns from df
df = df[[fname]]

for file in filepaths[1:]:
    # get name for column
    fnames = os.path.splitext(os.path.basename(file))[0]
    fname = fnames.replace(',', '-')
    
    # open file as Pandas DF
    temp_df = pd.read_csv(file,
                          skiprows=98,
                            index_col=['Raman Shift'],
                            usecols=['Raman Shift',
                                     'Dark', 
                                     'Raw data #1',
                                     'Dark Subtracted #1']
                            ).rename(columns={'Dark Subtracted #1' : fname})
    
    # Calculate uncorrected spectrum to show fluorescence
    temp_raw = pd.DataFrame({fname +'_raw' : temp_df['Raw data #1'] - temp_df['Dark']})

    # drop unnecessary columns from df
    temp_df = temp_df[[fname]]
    
    # now join on index
    df     = df.join(temp_df)
    df_raw = df_raw.join(temp_raw)


# rename to match dna data
df = df.rename(columns=lambda x: x.replace('SP_', 'UTK-'))
df_raw = df_raw.rename(columns=lambda x: x.replace('SP_', 'UTK-'))

# cut dfs to 400 since that gives best range
df = df[df.index > 450]
df_raw = df_raw[df_raw.index > 450]

# standardize all values to between 0 and 1 for both raw and baseline corrected
# also adjust to 0 as lowest value for later plotting
# first adjust up
df     = (df     - df.min()    )
df_raw = (df_raw - df_raw.min())

# then normalize
df     = df      / df.max()
df_raw = df_raw /  df_raw.max()

# add to DNA df to plot
data = dna.join(df.T, how='outer')

# copy over color, qty and allele on second instance (a) for 5 and 6
copy_cols = ['Color', 'Qty', 'Allele']

data.loc['UTK-5a', copy_cols] = data.loc['UTK-5', copy_cols]
data.loc['UTK-6a', copy_cols] = data.loc['UTK-6', copy_cols]

# only select those for which we have data
nacols = data.columns[3:]
data.dropna(subset=nacols, inplace=True) 

# Plot per color
ref   = data[data['Color'] == 'Reference'].index
brown = data[data['Color'] == 'Brown'].index
black = data[data['Color'] == 'Black'].index
calc  = data[data['Color'] == 'Calcined'].index
treat = data.filter(regex=(r'^(?!^UTK-[\d]*[\w]$).*$'), axis=0).index

ls = {'Reference': ref,
      'Thermally Altered' : brown,
      'Carbonized': black}

# plot the three levels we have
fig, (ax1, ax2) = plt.subplots(2,3, sharex=True, sharey=True, figsize=[10,5])

# set y labels along left y axis
ax1[0].set_ylabel('Adj. Intensity (a.u.)')
ax2[0].set_ylabel('Adj. Intensity (a.u.)')

# loop over ls to plot
for i, key in enumerate(ls):
    ax1[i].set_title(key, fontsize=10)
    ax1[i].plot(df[ls[key]], color='k', alpha=0.5)
    ax2[i].plot(df_raw[ls[key] + '_raw'], color='k', alpha=0.5)
    ax2[i].set_xlabel('Raman Shift (cm$^{-1}$)')
    
    ax1[i].autoscale(enable=True, axis='x', tight=True)
    ax1[i].set_ylim(-0.05, 1.15)
    
plt.savefig(outdir + '\\RamanComparisons.png', bbox_inches='tight', dpi=800)
plt.plot()
    
# plot treatments and compare to untreated to see if any differences arose
# all came from UTK-15
treatment_names = {'Soap'  : 'Soap',
                   'Bleach': 'bleach',                   
                   'Chloroform/MeOH': 'pretreatment_a', 
                   'NaOH' : 'pretreatment_b', 
                   'Combined Pretreatment' : pd.Index(['pretreatment_a_2',
                                              'pretreatment_b_2', 
                                              'pretreatment_c'])}

fig, (ax1, ax2) = plt.subplots(2,5, sharex=True, sharey=True, figsize=[15,5])

# 0 is bleach, 1 is soap, 2 is pre_a, 3 is pre_b, 4 is combpre
# set y labels along left y axis
ax1[0].set_ylabel('Adj. Intensity (a.u.)')
ax2[0].set_ylabel('Adj. Intensity (a.u.)')

# loop over ls to plot
for i, key in enumerate(treatment_names):
    ax1[i].set_title(key, fontsize=10)
    
    ax1[i].plot(df[treatment_names[key]], color='k', alpha=0.5)
    ax1[i].plot(df['UTK-15'], color='k', alpha=0.75)
    
    ax2[i].plot(df_raw[treatment_names[key] + '_raw'], color='k', alpha=0.5)
    ax2[i].plot(df_raw['UTK-15_raw'], color='k', alpha=0.8)
    
    ax2[i].set_xlabel('Raman Shift (cm$^{-1}$)')
    
    ax1[i].autoscale(enable=True, axis='x', tight=True)
    ax1[i].set_ylim(-0.05, 1.15)
    
plt.savefig(outdir + '\\RamanPretreatComparisons.png', bbox_inches='tight', dpi=800)
plt.plot()





