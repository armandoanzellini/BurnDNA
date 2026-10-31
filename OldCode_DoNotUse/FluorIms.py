# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 21:09:42 2025

@author: Armando Anzellini

!!! idea, what if just the fluorescence doesnt work, then maybe divide the fluor
image from the white light image and then try again (maybe divide the FFT)

Run in OpenCVenv

This file does does histograms and FFT PLSR
The other file pixel counts and FFT Power1D correlations.
"""
import os
import re
import glob
import cv2
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

foto_directory = "C:\\Users\\arman\\Dropbox\\Projects\\DNA Burned Remains\\Data\\Fluorescence Photos\\bl\\"
dna_directory  = "C:\\Users\\arman\\Dropbox\\Projects\\DNA Burned Remains\\Data\\"

dna_file = 'Pre-screening workbook.xlsx'

# get DNA data
cols = ['ID', 'Collection', 'Sample', 'Color', 'Qty','Allele']

dna = pd.read_excel(dna_directory + dna_file, sheet_name='ForCorr', skiprows=1, names=cols, index_col='Sample')

# remove SHSU samples since they don't have images
dna = dna[dna['Collection']!='SHSU']

# select only relevant columns from DNA data
dna = dna[['Color', 'Qty', 'Allele']].fillna(0)

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
        
    
    return gray.flatten(), ft.flatten(), hist


# make dictionaries to fill with data then make into pandas df
im_dict   = {}
ft_dict   = {}
hist_dict = {}

seen = {} # this is only to make sure all the images are lodaed even if multiple frags

for file in file_paths:
    # get filename for column
    fname = os.path.splitext(os.path.basename(file))[0]
    
    # get only the sample number to name it as the ID
    fid = re.search(r'([a-zA-Z ]*\d+)', fname).group()
    
    # get flattened fourier transform array of image
    im, fourier, hist = get_flat_fft(file)
    
    if fid in seen:
        seen[fid] += 1
        idn = fid + "." + ("I" * seen[fid])
        im_dict[idn] = im
        ft_dict[idn] = fourier
        hist_dict[idn] = hist
    else:
        seen[fid] = 0
        im_dict[fid] = im
        ft_dict[fid] = fourier
        hist_dict[fid] = hist
        

# make DataFrames
df_im   = pd.DataFrame.from_dict(im_dict)
df_ft   = pd.DataFrame.from_dict(ft_dict)
df_hist = pd.DataFrame.from_dict(hist_dict)

# fit a gaussian on the histograms DOES NOT WORK
# figure out where the max peak is located in each of the historgrams
# this will be mu for the gaussian
histloc = df_hist.idxmax() # max peak location is location of mu
histmu  = df_hist.max() # this will be max value gaussian
histsig = df_hist.std() # standard deviation

# get mean, sd, skew and kurtosis
h_mean = df_hist.mean().rename('Mean')
h_sd   = df_hist.std().rename('SD')
h_skew = df_hist.skew().rename('Skew')
h_kurt = df_hist.kurtosis().rename('Kurt')


# add those values as columns to dna creating a new df
imlst = list(h_sd.index)

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
dnadat = pd.DataFrame.from_dict(dnaim_dict, orient='index', columns = dna.columns)


# add all the histogram values
dnadat = dnadat.join([h_mean,
                      h_sd,
                      h_skew,
                      h_kurt])

# see if any values correlate to QTY and/or Allele
#plt.scatter(dna['Mean'], dna['Qty'], label='Mean') didn't work
plt.scatter(dna['SD'], dna['Qty'])
for idx, row in dna.iterrows(): 
    plt.text(row['SD'], row['Qty'], idx)
#plt.scatter(dna['Skew'], dna['Qty'], label='Skew') didn't work
#plt.scatter(dna['Kurt'], dna['Qty'], label='Kurt') didn't work
plt.title('Standard Deviation v. DNA Qty')
plt.ylabel('Qty')
plt.xlabel('Standard Deviation')
plt.show()

# remove outliers and try again
dna_clean = dna[dna['Qty']<5]

# fit a gaussian to the SD of the histogram
# define a gaussian function
def gauss(x, mu, sigma, a, b):
    g = (a/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2) + b
    return g

# define parameters for gaussian fit
dna_clean = dna_clean.sort_values(by='SD') # sort values by x values

dna_clean.drop(index=['Sample 9'], inplace=True) # drop outlier value in sample 9. could be the result of poor extraction or idiosyncrcies

x = np.array(dna_clean['SD'])
y = np.array(dna_clean['Qty'])
xmin, xmax = x.min(), x.max()
i_max = xmax-xmin  # index of highest value - for guess, assumed to be Gaussian peak
ymax = y.max()    # height of guessed peak
mu0 = i_max     # centre x position of guessed peak
b0 = 0  # height of baseline guess

# Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
# half, and even if this point is a distant outlier the fit should still converge.
sigma0 = (xmax-xmin)/4

a0 = (ymax - b0) * sigma0 * np.sqrt(2*np.pi) # estimate a, which is a scaling factor for height

# fit the curve
popt, _ = curve_fit(f=gauss, xdata=x, ydata=y, 
                                p0 = (mu0, sigma0, a0, b0),
                                bounds=((xmin, 0, 0, 0),
                                        (xmax, xmax - xmin, np.inf, y.max()),),
                                )
xtest = np.arange(-5, 5, 0.1)

param = sp.stats.t.fit([x,y], 17, loc=mu0, scale=sigma0/3)

pdf_fitted = sp.stats.t.pdf(x, loc=param[1], scale=param[2], df=param[0])

# interpolate for smooth function
xplot = np.arange(x.min(), x.max(), 0.001)

# adjusted parameters for better fit
p = np.array([0.23038166, 0.02282441, 0.13346411, 0.        ])

# plot scatter and gauss to check fit
plt.scatter(dna_clean['SD'], dna_clean['Qty'])
plt.plot(xplot, gauss(xplot, *p), color='orange')
plt.title('Standard Deviation v. DNA Qty')
plt.ylabel('Qty')
plt.xlabel('Standard Deviation')
plt.show()

# get 95% confidence interval and 99% confidence interval
z_99 = 2.576
z_95 = 2

def interval(mu, sigma, z, n):
    err  = z*sigma
    low  = mu - err
    high = mu + err
    
    return {'Margin of Error': err, 'Conf. Interval' : [low, high]}

mu, sigma = p[0:2]
n = len(dna_clean)

conf99 = interval(mu, sigma, z_99, n)
conf95 = interval(mu, sigma, z_95, n)

rang99 = xplot[np.where((xplot > conf99['Conf. Interval'][0]) & (xplot < conf99['Conf. Interval'][1]))]
rang95 = xplot[np.where((xplot > conf95['Conf. Interval'][0]) & (xplot < conf95['Conf. Interval'][1]))]

# plot scatter and gauss to check fit but with a good plot to use in pubs
# divide by color to plot better and change colors and shapes
dna_ref = dna_clean[dna_clean['Color']=='Reference']
dna_brw = dna_clean[dna_clean['Color']=='Brown']
dna_blk = dna_clean[dna_clean['Color']=='Black']
dna_cal = dna_clean[dna_clean['Color']=='Calcined']

plt.scatter(dna_ref['SD'], dna_ref['Qty'], label='Reference', marker = 'x', color='yellow')
plt.scatter(dna_brw['SD'], dna_brw['Qty'], label='Brown',marker = '*', color='brown')
plt.scatter(dna_blk['SD'], dna_blk['Qty'], label='Black',marker = '+', color='black')
plt.scatter(dna_cal['SD'], dna_cal['Qty'], label='Calcine',marker = 'o', color='gray')
plt.legend()

plt.plot(xplot, gauss(xplot, *p), color='orange')
# plt.fill_between(rang99,gauss(rang99, *p), color='red', alpha=0.5)
# plt.fill_between(rang95,gauss(rang95, *p), color='blue', alpha=0.5)
plt.title('Standard Deviation v. DNA Qty')
plt.ylabel('Qty')
plt.xlabel('Standard Deviation')
plt.vlines(.16, 0, 2.5)
plt.vlines(.31, 0, 2.5)
plt.show()

# checking allele accuracy with logit regression after looking at plots
plt.scatter(dna_ref['SD'], dna_ref['Allele'], label='Reference', marker = 'x', color='yellow')
plt.scatter(dna_brw['SD'], dna_brw['Allele'], label='Brown',marker = '*', color='brown')
plt.scatter(dna_blk['SD'], dna_blk['Allele'], label='Black',marker = '+', color='black')
plt.scatter(dna_cal['SD'], dna_cal['Allele'], label='Calcine',marker = 'o', color='gray')
plt.vlines(.16, 0, 100)
plt.vlines(.31, 0, 100)
for idx, row in dna_cal[dna_cal['SD'].between(.16, .31)].iterrows(): 
    plt.text(row['SD'], row['Allele'], idx)


plt.legend()


for col in dna.columns[3:]:
    plt.scatter(dna_clean[col], dna_clean['Allele'])
    plt.title(col + ' Allele')
    plt.show()

# Forget everything above, it does not help at all, so now try with the whole image
#!!! Use PCA or DFA now from FT images
# Add DNA QTY and Allele values to the data frame
imdat = df_im.T # transpose table of images and FT
ftdat = df_ft.T

dna_df = pd.DataFrame(columns=[['Qty', 'Allele']])

for i in imdat.index:
   samp = re.search('(\w* \d*)', i).group()
   dna_df.loc[i,'Qty'] = dna.loc[samp, 'Qty']
   dna_df.loc[i,'Allele'] = dna.loc[samp, 'Allele']
   
# add qty and allele columns to dataframes
imdat['Qty']    = dna_df['Qty'].astype('float64')
imdat['Allele'] = dna_df['Allele'].astype('float64')
ftdat['Qty']    = dna_df['Qty'].astype('float64')
ftdat['Allele'] = dna_df['Allele'].astype('float64')

# create function to loop through all 4 options
def run_PCR(data):
    y = data['Qty']
    X = data.drop(['Qty', 'Allele'], axis=1).astype('float64')
    
    pca = PCA()
    X_reduced = pca.fit_transform(scale(X))
    
    # 10-fold CV, with shuffle
    n = len(X_reduced)
    kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)
    
    regr = LinearRegression()
    mse = []
    
    # Calculate MSE with only the intercept (no principal components in regression)
    score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
    mse.append(score)
    
    # Calculate MSE using CV for the 19 principle components, adding one component at the time.
    for i in np.arange(1, 20):
        score = -1*model_selection.cross_val_score(regr, X_reduced[:,:i], y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
        mse.append(score)
        
    # Plot results    
    plt.plot(mse, '-v')
    plt.xlabel('Number of principal components in regression')
    plt.ylabel('MSE')
    plt.title('Salary')
    plt.xlim(xmin=-1)
    plt.show()
    
    pca2 = PCA()
    
    # Split into training and test sets
    X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=1)
    
    # Scale the data
    X_reduced_train = pca2.fit_transform(scale(X_train))
    n = len(X_reduced_train)
    
    # 10-fold CV, with shuffle
    kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)
    
    mse = []
    
    # Calculate MSE with only the intercept (no principal components in regression)
    score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
    mse.append(score)
    
    # Calculate MSE using CV for the 19 principle components, adding one component at the time.
    for i in np.arange(1, 20):
        score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:i], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
        mse.append(score)
    
    plt.plot(np.array(mse), '-v')
    plt.xlabel('Number of principal components in regression')
    plt.ylabel('MSE')
    plt.title('PCA after Split')
    plt.xlim(xmin=-1)
    plt.show()
    
    X_reduced_test = pca2.transform(scale(X_test))[:,:7]
    
    # Train regression model on training data 
    regr = LinearRegression()
    regr.fit(X_reduced_train[:,:4], y_train)
    
    # Prediction with test data
    pred = regr.predict(X_reduced_test)
    mean_squared_error(pred, y_test)
    
    plt.scatter(y_test, pred)
    plt.show()

for i in [dna.drop('Color', axis=1)]:
    run_PCR(i)


# alleles v. SD actually create a good range for extraction see plot above
"""
Instead of getting fancy, first try:
    1) Look to see if theres a correlation between location of largest peak and qnty
    2) Use location and QNTY to get maybe a logistic regression? or regular regression (non-linear)
    
    !!! tested this and there is no correlation at all. See code below
    # figure out where the max peak is located in the historgram
    histloc = df_hist.idxmax()

    # write to excel
    # histloc.to_excel(dna_directory + 'HistLoc.xlsx')

    # used Excel to make sure all Sample numbers matched the qty and allele even if duplicated
    # re-opened file to correlate
    hist_corr = pd.read_excel(dna_directory + 'HistLoc.xlsx')
    
    These simple solutions, if working, may not be as robust but would be faster
    If needed, more robust could be found, but first let's try simple
"""
# do PLSR on FT and Regular images
data = dna.join(df_ft.T)

data = data[data['Qty']<6] # remove outliers

X = np.log(abs(data.iloc[:,8:])).to_numpy() # getting absolute due to complex values
y = data['Qty']


kf = KFold(n_splits=10, shuffle=True, random_state=42)

scores = {}
for i in range(10):
    pls = PLSRegression(n_components=i+1)
    pls.fit(X, y)
    scores[i+1] = np.mean(-cross_val_score(pls, X, y, cv=kf, scoring='neg_mean_squared_error'))
    
plt.plot(scores.keys(), scores.values())


x_train, x_test, y_train, y_test = train_test_split(X, y)
# figure out best number of components

pls = PLSRegression(n_components = 4)
pls.fit(x_train, y_train)

plt.scatter(pls.predict(x_train), y_train, alpha=0.5, color='k')
plt.scatter(pls.predict(x_test), y_test, alpha=0.5, color='orange')
plt.title('PLS Regression FT Image on Qty')
plt.xlabel('Predicted Value')
plt.ylabel('Real Value')
# plt.savefig(rf'C:\Users\ara622\Desktop\Flourescence Figures\PLS_Qty')
plt.show()
