############## Importing Libraries ########################
#Plotting 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.lines as mlines
import seaborn as sns

#pandas and numpy dataframe manipulation 
import pandas as pd
import numpy as np

#datetime modules for sampling times 
from datetime import datetime
from datetime import timedelta

#scikit learn modules for modelling 
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler, 
    Normalizer)
from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_predict

#scipy modules for statistics, spectra normalisation, ALS baseline estimation 
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import cholesky
from scipy.stats import norm
from scipy.sparse.linalg import spsolve

################ Read in the Data ##########################
reference_raman_spectra= pd.read_csv("./data/Raw Raman Data - Reference.csv", index_col=0)
reference_raman_spectra.columns.name="Concentration IgG 1 (mg/mL)"
reference_raman_spectra.head(n=20)

experimental_raman_spectra= pd.read_csv("./data/Raw Raman Data - Experiment.csv").T #csv of experimental data must be transposed
experimental_raman_spectra.columns=experimental_raman_spectra.iloc[0] #Set the first row as the column header.
experimental_raman_spectra.drop(index=experimental_raman_spectra.index[0], inplace=True) #Drop the first row
experimental_raman_spectra.index=pd.Index(experimental_raman_spectra.index.astype(int), dtype='int64', name='RamanShift') 
sampling_time = [timedelta(minutes=minutes) for minutes in experimental_raman_spectra.columns]
sampling_time_formatted = [f"{(time.seconds//60):02}:{(time.seconds%60):02}" for time in sampling_time]
experimental_raman_spectra.columns=sampling_time_formatted
experimental_raman_spectra.columns.name="Sampling Time (MM:SS)"
experimental_raman_spectra.head(n=20)

############################### Define Plotting Functions ###################################  

# Define the variables that will be used in the plotting funtions: 
wavenumbers = np.sort(np.array(experimental_raman_spectra.index, dtype=int)) #Need to sort the wavenumbers for the x-axis
purple_colors = cm.viridis(np.linspace(0, 1, len(reference_raman_spectra.columns))) #Generate a shorter colourmap of the viridis colours to cycle through. 
red_colors = cm.seismic(np.linspace(0, 1, len(experimental_raman_spectra.columns)))

#Funtion to plot the raw data, before pre-processing steps: 
def plot_raw_spectra(dataframe, colorscheme):
        fig, ax = plt.subplots()
        for i, column in enumerate(dataframe.columns):
            ax.plot(
                dataframe.index,
                dataframe[column],
                label=column,
                color=colorscheme[i]
            )
        ax.set_xticks(wavenumbers[::200])
        ax.set_xticklabels(wavenumbers[::200], rotation=45, 
                        fontdict= {'fontsize': 'x-small',}
                        )
        ax.set_xlim(0,3300)
        ax.set_ylim(2500,0.5e6)
        ax.set_xlabel(r'Raman Shift (cm$^{-1})$')
        ax.set_ylabel("Raman Intensity (a.u.)")
        ax.set_title("Reference Raman Spectra",
                    fontdict = {'fontsize': 'large','fontweight' : "bold",}
                    )
        ax.legend(title= dataframe.columns.name,
          loc="upper left",
          bbox_to_anchor=(1.05, 1),
          borderaxespad=0
          )

#Funtion to plot the raw data overlaid spectra, before pre-processing steps: 
def plot_overlaid_spectra(dataframe_1, colorscheme_1, dataframe_2, colorscheme_2):
        fig,ax=plt.subplots()
        for i, column in enumerate(dataframe_1.columns):
            ax.plot(
                dataframe_1.index,
                dataframe_1[column],
                label=column,
                color=colorscheme_1[i],
                linestyle="--"
            )
        for i, column in enumerate(dataframe_2):
            ax.plot(
                dataframe_2.index,
                dataframe_2[column],
                label=column,
                color=colorscheme_2[i]
            )
        ax.set_xticks(wavenumbers[::200])
        ax.set_xticklabels(wavenumbers[::200], rotation=45, 
                        fontdict= {'fontsize': 'x-small',}
                        )
        ax.set_xlim(300,3300)
        ax.set_ylim(2500,0.5e6)
        ax.set_xlabel(r'Raman Shift (cm$^{-1})$')
        ax.set_ylabel("Raman Intensity (a.u.)")
        ax.set_title(f"overlay with reference spectra",
                    fontdict = {'fontsize': 'large','fontstyle' : "italic",}
                    )
        fig.suptitle(f"Experimental Raman Spectra",
                    fontweight="bold"
                    )
        #Generate the legends: 
        handles, labels = ax.get_legend_handles_labels()
        legend_2= ax.legend(handles[len(dataframe_1.columns):], 
                            labels[len(dataframe_1.columns):],
                            title=dataframe_2.columns.name,
                            ncols=2,
                            loc="upper left",
                            bbox_to_anchor=(1, 0.5),
                            frameon=False
                            )
        ax.add_artist(legend_2)
        ax.legend(handles[0:len(dataframe_1.columns)], 
                  labels[0:len(dataframe_1.columns)],
                  title=dataframe_1.columns.name,
                  loc="upper left",
                  bbox_to_anchor=(1, 1),
                  frameon=False
                )

############################### Preprocessing Steps ###################################
"""
These steps will involve: 
1. Applying a first derivative to the spectra
2. Normalising (sacaling) the data. 
    - A plot of 3 different normalisation techniques side by side will be prepared to judge the most effective approach to scaling for our data. 
3. Baseline Subtraction using the APLS algorithm for baseline correction
4. Identification of the peak of interest from the 1st derivative, normalised, baseline corrected data. 
5. A final baseline correction, concentrated on the region of interest in the spectra 
"""
#1:Compute the first derivative using the Savitzky-Golay filter, and 
#2: Apply the scaling to the first derivative
def normalised_first_derivative(dataframe, scaler):  
    first_der = savgol_filter(np.array(dataframe), window_length=5, delta=dataframe.index[1] - dataframe.index[0], polyorder=3)
    scaled_spectra = scaler.fit_transform(first_der)  # Reshape to 2D to avoid errors with this function
    return scaled_spectra

scaled_data = {} #initialise an empty dictionary to save the scaled data to
# Iterate over scalers and spectra to populate the scaled_data dictionary
for scaler, scaler_name in zip([preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.MaxAbsScaler()], ["SNV", "minmax", "maxabs"]):
    for spectra, spectra_name in zip([reference_raman_spectra, experimental_raman_spectra], ['reference', 'experimental']):
        scaled_spectra = normalised_first_derivative(spectra, scaler)
        key_name = f"{scaler_name}_scaled_{spectra_name}_spectra" 
        scaled_data[key_name] = scaled_spectra
#Plot the different normalisation techniques
def plot_normalisation_techniques():
    fig,ax=plt.subplots(1, 3, figsize=(19, 5), layout = "constrained")

    for i, scaler in enumerate(["SNV", "minmax", "maxabs"]):
        reference_data_scaled = scaled_data[f"{scaler}_scaled_reference_spectra"]
        reference_data_scaled = pd.DataFrame(
            data= scaled_data[f"{scaler}_scaled_reference_spectra"],
            index=reference_raman_spectra.index,
            columns=reference_raman_spectra.columns
        )
        experimental_data_scaled = pd.DataFrame(
            data= scaled_data[f"{scaler}_scaled_experimental_spectra"],
            index=experimental_raman_spectra.index,
            columns=experimental_raman_spectra.columns
        )
        #Plot the reference spectra:
        for j, column in enumerate(reference_data_scaled):
            ax[i].plot(
                reference_raman_spectra.index,
                reference_data_scaled[column],
                linestyle="--",
                color=purple_colors[j],
                label=f"Reference {column}")
        #Plot the experimental data: 
        for j, column in enumerate(experimental_data_scaled):
            ax[i].plot(
                experimental_data_scaled.index,
                experimental_data_scaled[column],
                linestyle="-",
                color=red_colors[j],
                label=f"Experimental {sampling_time_formatted[j]}")
        ax[i].set_xlim(300,2000)
        if scaler == "SNV":
            ax[i].set_ylim(-0.6, 0.2)
        elif scaler == "minmax":
            ax[i].set_ylim(0, 0.2)
        else:
            ax[i].set_ylim(0, 0.2) 
        ax[i].set_xlabel(r'Raman Shift (cm$^{-1})$')
        ax[i].set_ylabel(f"{scaler} Normalised Raman Intensity (a.u.)")
        ax[i].set_title(f"{scaler} Scaled 1st Der Data",fontdict = {'fontsize': 'large','fontstyle' : "italic",})
        
    fig.suptitle(f"Reference and Experimental Raman Spectra Overlay",
                fontweight="bold",
                fontsize="xx-large"
                )
    #Generate the legends: 
    handles, labels = ax[2].get_legend_handles_labels()
    exp_data_legend=ax[2].legend(handles=handles[len(reference_raman_spectra.columns):], 
                            labels=sampling_time_formatted,
                            title="Experimental Spectra (Time MM:SS)",
                            loc="upper left",
                            bbox_to_anchor=(1, 0.6),
                            ncols=2,
                            frameon=False)
    ax[2].add_artist(exp_data_legend)
    ax[2].legend(handles[0:len(reference_raman_spectra.columns)], 
            labels[0:len(reference_raman_spectra.columns)],
            title="Reference Spectra (mg/mL IgG conc.)",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False
            )

#3: Baseline correction
def als(y, lam=1e6, p=0.1, itermax=10): #This function computes the baseline to be subtracted from the original spectra
    L = len(y)
    D = sparse.eye(L, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]
    D = D.T
    w = np.ones(L)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def subtract_als_baseline(scaler_name, spectra_name): #This function subtracts the als basleine computed in the als algorithm above
    spectra=scaled_data[f"{scaler_name}_scaled_{spectra_name}_spectra"]
    als_baseline = np.array([als(spectra[:, i]) for i in range(spectra.shape[1])]).T
    return(spectra-als_baseline)

#A plot to visualise the baseline corrected spectra next to the original 
def plot_baseline_corrected_spectra(scaler_name, spectra_name):
    baseline_corrected_spectra = subtract_als_baseline(scaler_name, spectra_name)
    fig, ax = plt.subplots()
    for i in range(baseline_corrected_spectra.shape[1]):
        ax.plot(reference_raman_spectra.index, baseline_corrected_spectra[:,i], color="orange")
    ax.plot([], [], label="Baseline Corrected", color="orange")  #Want one label for the baseline corrected and one label for the original
    for i in range(baseline_corrected_spectra.shape[1]):
        ax.plot(reference_raman_spectra.index, scaled_data[f"{scaler_name}_scaled_{spectra_name}_spectra"][:,i], color="blue")
    ax.plot([], [], label="Original", color="blue")
    ax.set_xlim(300,2000)
    ax.set_ylim(-0.5, 0.25)
    ax.set_xlabel(r'Raman Shift (cm$^{-1})$')
    ax.set_ylabel(f"SNV Normalised Raman Intensity (a.u.)")
    ax.set_title(f"Baseline Corrected {spectra_name} Spectra vs Original",fontdict = {'fontsize': 'large','fontstyle' : "italic",})   
    fig.suptitle(f"{spectra_name} Raman Spectra",
                fontweight="bold",
                fontsize="xx-large"
                )
    ax.legend()

#4: Identify peak/ region of interest in spectra:
def plot_peak_of_interest(dataframe_1, dataframe_2, xlim, ylim):
    fig, ax = plt.subplots()

    for i, column in enumerate(dataframe_1.columns):
        ax.plot(
            dataframe_1.index,
            (subtract_als_baseline("SNV","reference")).T[i],
            label=column,
            color=purple_colors[i],
            linestyle="--"
        )
    for i, column in enumerate(dataframe_2.columns):
        ax.plot(
            dataframe_2.index,
            (subtract_als_baseline("SNV","experimental")).T[i],
            label=column,
            color=red_colors[i]
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'Raman Shift (cm$^{-1})$')
    ax.set_ylabel("Raman Intensity (a.u.)")
    ax.set_title(f"overlay with reference spectra",
                fontdict = {'fontsize': 'large','fontstyle' : "italic",}
                )
    fig.suptitle(f"Experimental Raman Spectra",
                fontweight="bold"
                )
    #Generate the legends: 
    handles, labels = ax.get_legend_handles_labels()
    legend_2= ax.legend(handles[len(dataframe_1.columns):], 
                        labels[len(dataframe_1.columns):],
                        title=dataframe_2.columns.name,
                        ncols=2,
                        loc="upper left",
                        bbox_to_anchor=(1, 0.5),
                        frameon=False
                        )
    ax.add_artist(legend_2)
    ax.legend(handles[0:len(dataframe_1.columns)], 
                labels[0:len(dataframe_1.columns)],
                title=dataframe_1.columns.name,
                loc="upper left",
                bbox_to_anchor=(1, 1),
                frameon=False
            )





