# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:54:16 2025

@author: ara622

H2O bands: 1800-1950; 1350-1425
All powdered
"""
import os
import glob
import numpy as np
import pandas as pd
import scipy as sp
import arviz as az
import pymc as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression


direct = r"C:\Users\arman\Dropbox\Projects\DNA Burned Remains\Data\Spectral Data\CSV"
outdir = r"C:\Users\arman\OneDrive\Documents\AuthorPapers (in progress)\Burn DNA\Submission\Figures"

filepaths = glob.glob(direct + '\\*.csv')

# import csv
# direct = r"C:\Users\ara622\Dropbox\Projects\DNA Burned Remains\Data\Spectral Data"
# filepathsed = glob.glob(direct + '\\*.sed')
# def sed_to_csv(filepath):
#     # get filename and path from sed file to use as CSV filename and path
#     fnameext = os.path.basename(filepath)
#     fname,_  = os.path.splitext(fnameext)
#     path     = os.path.dirname(filepath)
    
#     # read binary sed file
#     with open(filepath, "r") as file:
#         sedfile = file.read()
    
#     # write to a CSV for ease of reading later
#     with open(path +'\\CSV' + f"\\{fname}.csv", 'w', newline='') as csvfile:
#         file = csv.writer(csvfile, delimiter=',')
#         dat = sedfile.split(sep='\n')
#         rows = [i.split(sep='\t') for i in dat]
#         file.writerows(rows)

# for file in filepathsed:
#     sed_to_csv(file)

# read the first file with the file title and number as column name
fnames = os.path.splitext(os.path.basename(filepaths[0]))[0].split(sep='_')
fname = 'UTK-' + fnames[-1].lstrip('0')

# when reading index is wavelength in nm and value is Reflect. %
df_raw = pd.read_csv(filepaths[0], skiprows=27, names=[fname])

for file in filepaths[1:]:
    # get name for column
    fnames = os.path.splitext(os.path.basename(file))[0].split(sep='_')
    fname = 'UTK-' + fnames[-1].lstrip('0')
    
    # open file as Pandas DF
    temp_df = pd.read_csv(file, skiprows=27, names=[fname])
    
    # now join on index
    df_raw = df_raw.join(temp_df)

df_raw.dropna(inplace=True) # drop NaNs row that shows up

# Convert to absorbtion as per Ryder et al 2026
df_abs = np.log10(1/(df_raw/100))

# upload DNA data
dna_directory  = "C:\\Users\\arman\\Dropbox\\Projects\\DNA Burned Remains\\Data\\"

dna_file = 'Pre-screening workbook.xlsx'

# get DNA data
cols = ['ID', 'Collection', 'Sample', 'Color', 'Qty','Allele']

dna = pd.read_excel(dna_directory + dna_file, sheet_name='ForCorr', skiprows=6, names=cols, index_col='ID')

# select only relevant columns from DNA data
dna = dna[['Color', 'Qty', 'Allele']].fillna(0)

# do Multiplicative scattering correction
# MSC= msc()
# MSC.fit(df_raw)
# df = MSC.transform(df_raw)

# cut out first few wavenumbers
df_abs = df_abs.iloc[3:]

# # do Standard normal variate
df_snv = (df_abs - df_abs.mean())/df_abs.std()


# plot hyperspectra
# for col in df_raw.columns:
#     df_raw[col].plot()
#     plt.title(col)
#     plt.show()
    
# Get second (df2) derivative using savitsky-golay
df2 = df_abs.apply(signal.savgol_filter, args=(31, 3), deriv=2) # from Ryder et al

# plot to check
df_abs.plot(legend=False)
plt.xlim(1000, 2500)
plt.ylim(0, 1.5)
plt.title('Original')
plt.show()

df2.plot(legend=False)
plt.xlim(1000, 2500)
plt.ylim(df2.min().min()-5e-5, 0.0005)
plt.title('Second Derivative')
plt.show()

# cut out UV and Visible, keeping only the IR range
# this is because the UV has very little variation and it also leads to a 
# major peak in the 2nd derivative
# the visible is only picking up color differences in the bone so it sorts by color
ref_cut = df_raw[df_raw.index.to_series().between(1000, 2500)]/100
abs_cut = df_abs[df_abs.index.to_series().between(1000, 2500)]
dev_cut = df2[df2.index.to_series().between(1000, 2500)]

# PLSR using 2nd derivative
data = dna.join(dev_cut.T) # join spectra to DNA data on Sample ID

# remove outliers
# data = data[data['Qty'] < 6]

# Plot per color
ref   = dna[dna['Color'] == 'Reference'].index
brown = dna[dna['Color'] == 'Brown'].index
black = dna[dna['Color'] == 'Black'].index
calc  = dna[dna['Color'] == 'Calcined'].index

ls = {'Reference': ref,
      'Thermally Altered' : brown,
      'Carbonized': black,
      'Calcined' : calc }

fig, (ax0, ax1, ax2) = plt.subplots(3, 4, sharex=True, sharey='row', figsize=(13,6.5))
ax0[0].set_ylabel('Reflectance')
ax1[0].set_ylabel('Absorbance')
    
for i, key in enumerate(ls):
    ax0[i].set_title(key)      
    
    # create mean and upper and lower ranges of reflect plot
    mean_ref = ref_cut[ls[key]].mean(axis=1)
    upp_ref  = ref_cut[ls[key]].max(axis=1)
    low_ref  = ref_cut[ls[key]].min(axis=1)    
    
    # create mean and upper and lower ranges of abs plot
    mean_plot = abs_cut[ls[key]].mean(axis=1)
    upp_plot  = abs_cut[ls[key]].max(axis=1)
    low_plot  = abs_cut[ls[key]].min(axis=1)
    
    # create mean and upper and lower ranges of dev_2 plot
    mean_dev = dev_cut[ls[key]].mean(axis=1)
    upp_dev  = dev_cut[ls[key]].max(axis=1)
    low_dev  = dev_cut[ls[key]].min(axis=1)
    
    #ax0[i] plots the reflectance data
    ax0[i].plot(ref_cut[ls[key]], color='gray', alpha=0.75)
    ax0[i].plot(mean_ref, linestyle='-', color='k')
    ax0[i].plot(upp_ref, linestyle='--', color='k', alpha=.75)
    ax0[i].plot(low_ref, linestyle='--', color='k', alpha=.75)
    ax0[i].fill_between(mean_ref.index, upp_ref, low_ref, color='gray', alpha=0.25)
    
    # ax1[i] plots the abs data
    ax1[i].plot(abs_cut[ls[key]], color='gray', alpha=0.75)
    ax1[i].plot(mean_plot, linestyle='-', color='k')
    ax1[i].plot(upp_plot, linestyle='--', color='k', alpha=.75)
    ax1[i].plot(low_plot, linestyle='--', color='k', alpha=.75)
    ax1[i].fill_between(mean_plot.index, upp_plot, low_plot, color='gray', alpha=0.25)
    
    # ax2[i] plots the 2nd derivative data
    ax2[i].plot(dev_cut[ls[key]], color='gray', alpha=0.75)
    ax2[i].plot(mean_dev, linestyle='-', color='k')
    
    # set plot limits and show label
    ax1[i].set_xlim(mean_plot.index.min(), mean_plot.index.max())
    ax1[i].tick_params(labelbottom=True) # since sharing must explicitly re-add
    
    ax2[i].set_ylim(-0.00055, 0.00045)
    ax2[i].set_xlabel('Wavelength (nm)')

plt.savefig(outdir + '\\NIRComparisons.png', bbox_inches='tight', dpi=800)
plt.show()



def run_PLS(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    pls = PLSRegression(n_components=2)
    pls.fit(X_train, y_train)
    
    # figure out best number of components
    
    plt.scatter(y_train, pls.predict(X_train), alpha=0.5, color='k')
    plt.scatter(y_test, pls.predict(X_test), alpha=0.5, color='orange')
    plt.plot(np.linspace(0, y.max(), 50), np.linspace(0, y.max(), 50), linestyle='--', color = 'k')
    plt.title('Whole Spectrum 2nd Derivative PLS: ' + name)
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    
    plt.show()

# Bayesian Lasso regression on qty and a Bayesian classification on allele recovery
# the cut spectrum as before with a Bayesian lasso on qty regression

# Bayes Lasso
# to fix the arviz break without much hassle
def bayes_r2_score(y_true, y_pred):
    """
    Bayesian R2 per Gelman et al. 2019, replicating the old az.r2_score.
    y_true : array-like, shape (n_outputs,)
    y_pred : array-like, shape (n_posterior_samples, n_outputs)
    Returns pd.Series with 'r2' and 'r2_std', matching the old return signature.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    var_y_est = y_pred.var(axis=1)
    var_e = (y_true - y_pred).var(axis=1)
    r_squared = var_y_est / (var_y_est + var_e)
    return pd.Series([np.mean(r_squared), np.std(r_squared)], index=['r2', 'r2_std'])

def bayes_predict_lasso(X, y, n_components=0, dim_reduction=False, plot=True, savefig=False, test_size=0.1):
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Preprocessing: Dimensionality reduction if indicated
    if dim_reduction:
        n_components = n_components
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_train)
        
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_reduced)
        
    else:    
        pca = None
        n_components = X_train.shape[1]
        X_reduced = X_train
        
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_reduced)
    
    # Build Bayesian regression model
    with pm.Model() as model:
        # alpha and sigma parameter priors
        alpha = pm.Normal('alpha', mu=np.mean(y_train), sigma=np.std(y_train))
        sigma = pm.HalfNormal('sigma', sigma=np.std(y_train))
        
        # using sparsity as presented in:
            #Castillo et al 2015 and Carvahlo et al 2009
            # this changes the laplace of Bayes LASSO with a horseshoe dist
            #defined with two shirnking parameters in half cauchy applied
            # to the beta parameter
            
        # Global shrinkage parameter
        tau = pm.HalfCauchy("tau", beta=1)
        
        # Local shrinkage parameters (vector)
        lambda_ = pm.HalfCauchy("lambda_", beta=1, shape=n_components)
        
        # Regularized horseshoe prior factor (Piironen & Vehtari 2017)
        c2 = pm.InverseGamma("c2", alpha=1, beta=1)
        
        # Centered parametrization creates strong correlations between the 
        # hierarchical levels (beta depends directly on tau and lambda_), 
        # making it hard for the sampler to explore efficiently. The sampler 
        # gets stuck in the narrow part of the funnel.
        # So using non-centered parametrization of the        
        # Horsehoe prior to make this a Bayesian Lasso (updated (tm))
        
        # Regularized shrinkage
        lambda_tilde = (c2 * lambda_**2) / (c2 + tau**2 * lambda_**2)
        
        beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=n_components) # explore space
        beta = pm.Deterministic('beta', beta_raw * tau * pm.math.sqrt(lambda_tilde)) # regularized transformation

        # now set the linear regression function
        mu = alpha + pm.math.dot(X_scaled, beta)
        
        # and prior for the observations
        y_obs = pm.Normal(
                'y_obs',
                mu=mu,
                sigma=sigma,
                observed=y_train
                )

        # Run MCMC using parameters defined in CPU
        trace = pm.sample(
            2000,
            tune=1000,
            chains=5,
            cores=5,
            nuts_sampler='numpyro',
            target_accept=0.99,
            return_inferencedata=True
        )

    
    
    print("\n=== Model Summary ===")
    print(az.summary(trace, var_names=['alpha', 'sigma', 'tau'], ci_kind='hdi', ci_prob=0.94))
    
    print("\n=== Convergence Diagnostics ===")
    summary = az.summary(trace, var_names=['alpha', 'sigma', 'tau'], ci_kind='hdi', ci_prob=0.94)
    print(f"Max R-hat: {summary['r_hat'].max()}")
    print(f"Mean acceptance rate: {trace.sample_stats['acceptance_rate'].mean().values:.3f}") 
    print(f"\nNumber of divergences: {trace.sample_stats['diverging'].sum().values}")
    print(f"Min Effective Sample Size: {summary['ess_bulk'].min():.0f}")
    
    
    # Pre-extract and prepare posterior samples for prediction
    alpha_samples_all = trace.posterior['alpha'].values.flatten()
    beta_samples_all = trace.posterior['beta'].values.reshape(-1, n_components)
    sigma_samples_all = trace.posterior['sigma'].values.flatten()
    n_posterior_samples = len(alpha_samples_all)
    
    # Prediction function for new spectra
    def predict(new_spectra, n_samples=100):
        # Ensure n_samples doesn't exceed available samples
        n_samples = min(n_samples, n_posterior_samples)
        
        if dim_reduction and pca is not None:
            new_reduced = pca.transform(new_spectra)
        else:
            new_reduced = new_spectra
        
        new_scaled = scaler_X.transform(new_reduced)
        
        # Use pre-extracted samples (no need to extract every time)
        alpha_samples = alpha_samples_all[:n_samples]
        beta_samples = beta_samples_all[:n_samples]
        sigma_samples = sigma_samples_all[:n_samples]
        
        mu = alpha_samples[:, None] + (beta_samples @ new_scaled.T) #vectorized
        predictions = np.random.normal(mu, sigma_samples[:, None])
        
        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'lower': np.quantile(predictions, 0.025, axis=0),
            'upper': np.quantile(predictions, 0.975, axis=0),
            'samples': predictions
        }
    
    # Evaluate on TRAINING data
    train_predictions = predict(X_train)
    mae_train         = np.mean(np.abs(train_predictions['mean'] - y_train))
    r2_train          = bayes_r2_score(y_train, train_predictions['samples'])
    print("\n=== Training Set Predictions ===")
    print(f"Mean absolute error: {mae_train:.3f}")
    print(f"Within 95% credible interval: {np.mean((y_train >= train_predictions['lower']) & (y_train <= train_predictions['upper'])):.2%}")
    print(f"R2: \n{r2_train}")
    
    # Evaluate on TEST data
    test_predictions = predict(X_test)
    mae_test         = np.mean(np.abs(test_predictions['mean'] - y_test))
    r2_test          = bayes_r2_score(y_test, test_predictions['samples'])
    print("\n=== Test Set Predictions ===")
    print(f"Mean absolute error: {mae_test:.3f}")
    print(f"Within 95% credible interval: {np.mean((y_test >= test_predictions['lower']) & (y_test <= test_predictions['upper'])):.2%}")
    print(f"R2: \n{r2_test}")
    
    # Get Beta mean and 94% credibility interval to check that they don't cross 0
    beta_summary = az.summary(trace, var_names=['beta'], ci_kind='hdi', ci_prob=0.94)
    
    beta_mean  = [float(x) for x in beta_summary['mean'].values]
    beta_hdi3  = [float(x) for x in beta_summary['hdi94_lb'].values]
    beta_hdi97 = [float(x) for x in beta_summary['hdi94_ub'].values]
    
    
    low_err = np.array(beta_mean) - np.array(beta_hdi3)
    upp_err = np.array(beta_hdi97) - np.array(beta_mean)
    
    if plot:
        # plot the trace to see convergence
        az.plot_trace(trace,
                      var_names=['alpha', 'beta', 'sigma'],
                      figure_kwargs={'figsize': (10, 6)}
                      )
        
        # plot train and test data to check precision
        fig, ax = plt.subplots()
        
        # add the scatter plots of train and test data of predicted versus true
        ax.scatter(train_predictions['mean'], y_train,
                   alpha=0.7,
                   color='black')
        ax.scatter(test_predictions['mean'], y_test,
                   alpha=0.8,
                   color='orange')
        
        # add legend
        ax.legend(['Training Data', 'Testing Data'])
        
        # create a line of what the values should be 
        ax.plot(line_x := np.arange(y.min(), y.max(), 0.5), 
                line_y := np.arange(y.min(), y.max(), 0.5),
                linestyle='--',
                color='k')
        
        # create the encompasing error lines for MMD for that isotope
        # ax.plot(line_x, line_y + mmd,
        #         linestyle=':',
        #         color='k')
        # ax.plot(line_x, line_y - mmd,
        #         linestyle=':',
        #         color='k')
        # ax.fill_between(line_x, line_y + mmd, line_y - mmd,
        #                 alpha=0.4,
        #                 color='gray',
        #                 zorder=0)
        
        # add titles and axes
        # ax.set_title('IRIS ' + symbol + ' Bayesian Prediction')
        # ax.set_xlabel('Predicted ' + symbol + ' Values')
        # ax.set_ylabel('True ' + symbol + ' Values')
        
        plt.tight_layout()
        plt.show()
        
        # plot beta coefficients
        plt.figure(figsize=(10, 4))
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)  # Add zero line
        plt.errorbar(range(len(beta_mean)), beta_mean, 
                     yerr=(low_err, upp_err),
                     fmt = 'o',
                     capsize = 0.5)
        plt.xlabel('Feature Index')
        plt.ylabel('Beta Coefficient')
        plt.title('Bayes LASSO Results')
        plt.tight_layout()
        plt.show()
    
    
    return (summary['r_hat'].max(),
            mae_train,
            r2_train,
            mae_test,
            r2_test,
            beta_mean
           )


# Cutting out high QTY data
# qtydat = data[data['Qty'] < 6]
qtydat = data

# collagen peak ranges for second derivative (Ryder et al. 2026)
ch1_range   = (1672, 1703)
ch2_range   = (1717, 1748)
amI_range   = (2030, 2059)
amII_range  = (2155, 2195)
ch3_range   = (2250, 2300)

# find peaks within those ranges in the second derivative spectra
# peaks = {}

# # Specific peak locations
# for item in qtydat.T:
#     ipeaks = {}
#     i = qtydat.T[item]
    
#     ch1  = i.loc[ch1_range[0]:ch1_range[1]]
#     ch2  = i.loc[ch2_range[0]:ch2_range[1]]
#     ch3  = i.loc[ch3_range[0]:ch3_range[1]]
#     amI  = i.loc[amI_range[0]:amI_range[1]]
#     amII = i.loc[amII_range[0]:amII_range[1]]
    
#     ch1_ind, _  = sp.signal.find_peaks(-ch1, prominence=1e-7)
#     ch2_ind, _  = sp.signal.find_peaks(-ch2, prominence=1e-7)
#     ch3_ind, _  = sp.signal.find_peaks(-ch3, prominence=1e-7)
#     amI_ind, _  = sp.signal.find_peaks(-amI, prominence=1e-7)
#     amII_ind, _ = sp.signal.find_peaks(-amII, prominence=1e-7)
    
#     if len(ch1_ind)==0:
#         ipeaks['ch1'] = 0
#     else:
#         ipeaks['ch1']  = ch1.iloc[ch1_ind].values[0]
#     if len(ch2_ind)==0:
#         ipeaks['ch2'] = 0
#     else:
#         ipeaks['ch2']  = ch2.iloc[ch2_ind].values[0]
#     if len(ch3_ind)==0:
#         ipeaks['ch3'] = 0
#     else:
#         ipeaks['ch3']  = ch3.iloc[ch3_ind].values[0]
#     if len(amI_ind)==0:
#         ipeaks['amI'] = 0
#     else:
#         ipeaks['amI']  = amI.iloc[amI_ind].values[0]
#     if len(amII_ind)==0:
#         ipeaks['amII'] = 0
#     else:
#         ipeaks['amII'] = amII.iloc[amII_ind].values[0]     
    
#     peaks[i.name] = ipeaks
    
# peakdf = pd.DataFrame.from_dict(peaks, 
#                                 orient='index',
#                                 columns=['ch1', 'ch2', 'ch3', 'amI', 'amII'])


# General peak locations and height
peaks = {}
for item in qtydat.T:
    i = qtydat.T[item]
    
    peak_ind, _ = sp.signal.find_peaks(-i.iloc[3:], prominence=1e-6)
    
    peakarr = np.column_stack((i.index[peak_ind], i.iloc[peak_ind].values))
    
    peaks[item] = peakarr[peakarr[:,1] < 0]
    
    
peakdf = pd.DataFrame(data=np.float64(peaks['UTK-1'][:,1]),
                                      index=peaks['UTK-1'][:,0],
                                      columns=['UTK-1'])

for key in peaks.keys():
    if key == 'UTK-1':
        pass
    else:
        ipeakdf = pd.DataFrame(data=np.float64(peaks[key][:,1]),
                                              index=peaks[key][:,0],
                                              columns=[key])
        
        peakdf = peakdf.join(ipeakdf, how='outer')


# bin peaks into ranges where you find only a single peak or no peaks per item
def bin_peaks_df(df):
    bins = []
    current = [0]

    for i in range(1, len(df)):
        candidate_rows = current + [i]
        col_counts = df.iloc[candidate_rows].notna().sum(axis=0)
        single_peak_ok = (col_counts <= 1).all()

        if single_peak_ok:
            current.append(i)
        else:
            bins.append(current)
            current = [i]

    bins.append(current)

    records = []
    for bin_rows in bins:
        sub = df.iloc[bin_rows]
        median_index = np.median(sub.index)
        values = sub.median(axis=0)
        records.append(pd.Series(values, name=median_index))

    result = pd.DataFrame(records)
    result.index.name = df.index.name
    return result
   
# ensure that any peaks that appear above 0 are turned to 0 since non-existent
peaksdf = peakdf.where(peakdf < 0, 0)

# create data structure to analyze
# X = data.iloc[:,3:].to_numpy()
X = peaksdf.to_numpy()

y = qtydat['Qty'].to_numpy()

# Standardize
# X = (X - X.mean(0)) / X.std(0, ddof=1)
# y = (y - y.mean()) / y.std(ddof=1)


# get binned peaks
binnedpeaks = bin_peaks_df(peakdf)

peaksdf = binnedpeaks.fillna(0)  # set NaNs to 0

# using the binned peaks all
X = peaksdf.T.to_numpy()
y = data['Qty'].to_numpy()

n, p = X.shape


# run_PLS(X, y, 'Qty')

# for i in range(1, 20):
#     rhat, mae_train, r2_train, mae_test, r2_test, betas = bayes_predict_lasso(X, y,
#                                                     n_components=i,
#                                                     dim_reduction=True,
#                                                     plot=True,
#                                                     test_size=0.1)

# Could try peak ratios like with IRIS


y = qtydat['Allele'].to_numpy()
y_s = (y > 60).astype(int)


# regular logistic on Allele
# reduce shape prior to logit-reg
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(peaksdf.T.to_numpy())

# Split into training and testing
X_train = X_reduced
X_test = X_train
y_train = y_s
y_test = y_train


X_train, X_test, y_train, y_test = train_test_split(X, y_s, random_state=42)

log = LogisticRegressionCV(cv=5, 
                           solver='liblinear', 
                           random_state=42, 
                           fit_intercept=False,
                           use_legacy_attributes=True,
                           l1_ratios=(0,),
                           scoring='accuracy')

log.fit(X_train, y_train)

log_score = log.score(X_test, y_test)

# plot curve for results
line_x = np.linspace(x_min := 0, x_max := 1, num=50)

# plot
fig, ax = plt.subplots()

# plot binarized data and logistic function from prediction
ax.scatter(log.predict(X_train), y_train, c='k', alpha=0.5)
ax.scatter(log.predict(X_test), y_test, c='orange', alpha=0.5)
ax.plot(line_x, log.decision_function(line_x.reshape(-1,1)), 'k', linestyle = '--')


# trying random forest classifier and potentially a random forest regressor
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

rfc.predict(X_test)

rfc.score(X_test, y_test)

from sklearn.tree import plot_tree

plot_tree(rfc.estimators_[0], feature_names=peaksdf.index, class_names=['Accurate', 'Inaccurate'], filled=True, rounded=True)


# Bayesian classification method with large datasets


