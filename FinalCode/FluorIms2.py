# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 23:34:29 2025

@author: Armando Anzellini

!!! idea, what if just the fluorescence doesnt work, then maybe divide the fluor
image from the white light image and then try again (maybe divide the FFT)

Run in OpenCVenv

Did 1D Power Spectrum, now run just regular sum spectrum or mean spectrum and also try a Han window if not working perfectly

This file does pixel counts and FFT Power1D correlations. 
The other file does histograms and FFT PLSR

"""
import os
import re
import glob
import cv2
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

foto_directory = r"C:\Users\arman\Dropbox\Projects\DNA Burned Remains\Data\Fluorescence Photos\bl\\"
dna_directory  = r"C:\Users\arman\Dropbox\Projects\DNA Burned Remains\Data\\"

dna_file = 'Pre-screening workbook.xlsx'

# get DNA data
cols = ['ID', 'Collection', 'Sample', 'Color', 'Qty','Allele']

dna = pd.read_excel(dna_directory + dna_file, sheet_name='ForCorr', skiprows=1, names=cols, index_col='Sample')

# remove SHSU samples since they don't have images
dna = dna[dna['Collection']!='SHSU']

# select only relevant columns from DNA data
dna = dna[['Color', 'Qty', 'Allele']].fillna(0)

# make allele accuracy into a decimal to represent percentage
dna['Allele'] = dna['Allele']/100

file_paths = glob.glob(foto_directory + '\\*bl*.png')

def get_flat_fft(file, plot=False, save=False):
    # get image name
    name = os.path.basename(file).rstrip('.png')
    
    # read image
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    
    # crop the image to remove all white space
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray to make finding edges easier
    gray = 255*(gray < 200).astype(np.uint8) # To invert the image where white is now black ([0,0,0,0])
    coords = cv2.findNonZero(gray) # Find all non-zero points which is image
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    img = image[y:y+h, x:x+w] # Crop the image on the original image
    
    # plot to check; imshow requires that the rgba values be merged to show properly
    # b,g,r,a = cv2.split(img)
    # image_rgb = cv2.merge((r,g,b,a))
    # plt.imshow(image_rgb)
    # plt.show()
    
    # extract the alpha channel:
    a= img[:,:,3]
    
    #use that as a mask for bitwise compositing:
    img2 = cv2.bitwise_and(img,img,mask=a)
    
    #convert *that* to grayscale
    gray = cv2.cvtColor(img2,cv2.COLOR_BGRA2GRAY)
    
    # resize for consistent pixel sizes
    # following re-sizes it to 720p (960 pixels by 960 pixels)
    # some shrink and some upscale, so using the best algorithm for both
    gray = cv2.resize(gray, (960, 960), cv2.INTER_LANCZOS4)

    
    # scale up to ensure all images have the same maximum value and make equivalent
    # high = gray.max()
    # diff = 255 - high
    
    # gray += diff
    
    # gray[gray == diff] = 0 # return the black pixels to black after scaling up
    
    # get the histogram
    hist = cv2.calcHist([gray], channels=[0], mask=None, histSize=[100], ranges=[1,256])
    
    # flatten hist since that's how everything will take it and 2D makes no sense
    hist = hist.flatten()
    
    # normalize histogram by max so that all values go from 0 to 1
    # this makes PCA later easier and makes it easier to use the model on test images
    # equivalent to percentage of pixels in that bin
    hist = hist/hist.max()    
    
    # do FFT of the image
    ft = np.fft.fft2(gray) # do 2d fast fourier transform
    ft = np.fft.fftshift(ft) # have to shift 0,0 pixel to center
      
    
    # Try without FFT just on regular image? Read
    
    # 1d power spectrum of image
    fft_amplitudes = (np.abs(ft)**2) # get the frequency amplitudes
    
    # return a one dimensional array containing the wave vectors for the fft2
    freq = np.fft.fftfreq(960)*960 # convert to pixel frequency
    
    # since originally two dimensional, must get radial dimension
    freq2D = np.meshgrid(freq, freq)
    
    knrm = np.sqrt(freq2D[0]**2 + freq2D[1]**2) # norm of wave vectors
    
    # make spectrum norm 0 = 0, to remove the peak at 0,0
    fft_amplitudes[knrm == 0] = 0
    
    # flatten wavevectors and amplitudes
    knrm = knrm.flatten()
    fft_amplitudes = fft_amplitudes.flatten()
    
    #  create the number of bins
    kbins = np.arange(0.5, 960//2+1, 1.)
    
    # get midpoints of bins to sum
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    
    # compute mean fourier amplitude in each bin (try both mean and median)
    Abins, _, _ = sp.stats.binned_statistic(knrm, fft_amplitudes,
                                     statistic = "median",
                                     bins = kbins)
    
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    # create dataframe of power1d
    power1d = pd.DataFrame({ 'Wavevector': kvals, name : Abins} )
    
    if plot:
        # plot the gray image
        plt.imshow(gray, cmap='gray')
        plt.title(name + ' Grayscale')
        if save:
            plt.savefig(rf'C:\Users\arman\Desktop\Flourescence Figures\Per Sample\{name}')
            plt.close()
        else:
            plt.show()
        
        # plot FFT results
        plt.imshow(np.log(abs(ft)), cmap='gray')
        plt.title(name + ' FT')
        if save:
            plt.savefig(rf'C:\Users\arman\Desktop\Flourescence Figures\Per Sample\{name}_FT.png')
            plt.close()
        else:
           plt.show()
        
        # plot
        plt.loglog(kvals, Abins, color='k', linewidth=0.75)
        plt.title(name + ' 1D Power Spectrum')
        plt.xlabel('Wavevector')
        plt.ylabel('log(Amplitude)')
        if save:
            plt.savefig(rf'C:\Users\arman\Desktop\Flourescence Figures\Per Sample\{name}_Power1D.png')
            plt.close()
        else:
            plt.show()
        
    
    return gray.flatten(), ft.flatten(), power1d[name].to_numpy()

# make dictionaries to fill with data then make into pandas df
im_dict   = {}
ft_dict   = {}
pow_dict  = {}

seen = {} # this is only to make sure all the images are lodaed even if multiple frags

for file in file_paths:
    # get filename for column
    fname = os.path.splitext(os.path.basename(file))[0]
    
    # get only the sample number to name it as the ID
    fid = re.search(r'([a-zA-Z ]*\d+)', fname).group()
    
    # get flattened fourier transform array of image
    im, fourier, power1d = get_flat_fft(file, plot=True, save=False)
    
    if fid in seen:
        seen[fid] += 1
        idn = fid + "." + ("I" * seen[fid])
        im_dict[idn]  = im
        ft_dict[idn]  = fourier
        pow_dict[idn] = power1d

    else:
        seen[fid] = 0
        im_dict[fid]  = im
        ft_dict[fid]  = fourier
        pow_dict[fid] = power1d

        
# make DataFrames
df_im   = pd.DataFrame.from_dict(im_dict)
df_ft   = pd.DataFrame.from_dict(ft_dict)
df_pw   = pd.DataFrame.from_dict(pow_dict)


# create pixel value counts dataframe (only on sample indexed at 0)
px_count = df_im.iloc[:,0].value_counts().sort_index().to_frame(name=df_im.columns[0])

# rename index to clarify that it's pixel value not sample name
px_count.index.name = 'Value'

# loop through rest of columns (indexed 1 on) and append to df
for col in df_im.columns[1:]:
    # create temporary df of value counts for current column name
    tempdf = df_im[col].value_counts().sort_index().to_frame(name=col)
    
    # change the name of the index to match px_count
    tempdf.index.name='Value'
    
    # merge with px_count
    px_count = px_count.join(tempdf, how='outer')


# remove pixel value 0 row
px_count.drop(index=0, inplace=True)

# px_count.plot(legend=False)

# plotting shows 3 natural breaks: ~150, ~170, and potentially 200(?)
# try all three for percentages and run correlations on all
cor200 = px_count[px_count.index >= 200].sum()/px_count.sum()
cor170 = px_count[px_count.index >= 170].sum()/px_count.sum()
cor150 = px_count[px_count.index >= 150].sum()/px_count.sum()

# Now the DNA values need to be added to each of these in such a way that they
# match even if it's a duplicate, so anything before the period
imlst = list(px_count.columns)

# make a loop that creates a dictionary with the intended values and the
# cols as keys. That way they can be appended later
dnaim_dict = {}

for samp in list(dna.index):
    # regex function to find duplicate images
    r = re.compile(samp + r"(?=\.|$)")

    # find them in the list of images
    matches = list(filter(r.match, imlst))
    
    # put the DNA values into the dictionary for each image name match
    for i in matches:
        dnaim_dict[i] = dna.loc[samp].to_list()

# convert dictionary to dataframe with column names
cordna = pd.DataFrame.from_dict(dnaim_dict, orient='index', columns = dna.columns)

# make dummy variables for categorical analysis
cordna['Allele_Cat'] = [0 if a < .6 else 1 for a in cordna['Allele']]
cordna['Qty_Cat']    = [0 if a == 0.  else 1 for a in cordna['Qty']] 

# add pixel count percentage values to the cordna df
cordna = cordna.join([cor150.rename('PixPer150'),
                      cor170.rename('PixPer170'),
                      cor200.rename('PixPer200')])

corrdna = cordna[(cordna['Color'] != 'Reference') & (cordna['Color'] != 'Brown')]

# analyze after cutting out ref and brown
# plt.scatter(corrdna['PixPer200'], corrdna['Allele'])
# plt.show()

# plt.scatter(corrdna['PixPer200'], corrdna['Qty'])
# plt.show()

# Improve on FTT on images (have to respahe to original to plot)
# plt.imshow(np.log(abs(df_ft.iloc[:,0].to_numpy().reshape(960,960))), cmap='gray')
# plt.title(df_ft.iloc[:,0].name)

# # plot surface to see spectrum
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# xx, yy = np.mgrid[0:960, 0:960]
# ax.plot_surface(xx, 
#                 yy, 
#                 np.log(abs(df_ft.iloc[:,0].to_numpy().reshape(960,960))),
#                 cmap='viridis')
# ax.contour(xx,
#            yy, 
#            np.log(abs(df_ft.iloc[:,0].to_numpy().reshape(960,960))),
#            zdir='x', offset=0, cmap='coolwarm')
# ax.contour(xx,
#            yy, 
#            np.log(abs(df_ft.iloc[:,0].to_numpy().reshape(960,960))),
#            zdir='y', offset=960, cmap='coolwarm')
# ax.contour(xx,
#            yy, 
#            np.log(abs(df_ft.iloc[:,0].to_numpy().reshape(960,960))),
#            zdir='z', offset=0, cmap='gray')

# !!! Stop playing with it. Subtract the white light image from the blue light image in grayscale and output.
# !!! See above. Actually, white from fluor only shows underlying structure

pwdna = cordna.join(df_pw.T) # add power spectrum to DNA df to calculate correlations

# All of the brown and reference yielded, so let's check just carbonized and calcined
# pwdna = pwdna[(pwdna['Color'] != 'Reference') & (pwdna['Color'] != 'Brown')] # charred sample has qty of 11.5

colors = {'Reference' : 'darkkhaki', 'Brown' : 'peru', 'Black' : 'k', 'Calcined': 'gray'}

for burn, color in colors.items():
    dats = pwdna[pwdna['Color'] == burn]
    plt.loglog(dats.iloc[:,8:].T, color=color, alpha=0.5)

    plt.title(f'Image Power Spectra {burn}')
    plt.show()

# trying Bayesian approach


