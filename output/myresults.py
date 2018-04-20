#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cm
from numpy import *
from random import sample, seed, randint
from os.path import getsize as getFileSize
import math
import random
import csv
from cycler import cycler
from io import StringIO
#np.set_printoptions(threshold=np.nan)
from collections import Counter
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import AxesGrid
from astropy import units as u
from astropy import cosmology

import matplotlib.ticker as mtick
import PlotScripts
import ReadScripts
import AllVars
import ObservationalData as Obs

from mpi4py import MPI
from tqdm import tqdm

import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

AllVars.Set_Params_Kali()
AllVars.Set_Constants()
PlotScripts.Set_Params_Plot()


output_format = ".png"

# For the Tiamat extended results there is a weird hump when calculating the escape fraction.
# This hump occurs at a halo mass of approximately 10.3. 
# The calculation of fesc skips this hump range (defined from kink_low to kink_high)
kink_low = 10.3
kink_high = 10.30000001

m_low = 7.0 # We only sum the photons coming from halos within the mass range m_low < Halo Mass < m_high
m_high = 15.0

m_gal_low = 3.0 
m_gal_high = 12.0

m_low_SAGE = pow(10, m_low)/1.0e10 * AllVars.Hubble_h
m_high_SAGE = pow(10, m_high)/1.0e10 * AllVars.Hubble_h

bin_width = 0.2
NB = int((m_high - m_low) / bin_width)
NB_gal = int((m_gal_high - m_gal_low) / bin_width)

def raise_list_power(my_list, n):
    return [pow(x, n) for x in my_list]
def raise_power_list(my_list, n):
    return [pow(n, x) for x in my_list]

def calculate_beta(MUV, z):
    ''' 
    Calculation of the dust attenuation parameter Beta. Fit values are from Bouwens (2015) ApJ 793, 115.
    For z = 5 and 6, Bouwens uses a piece-wise linear relationship and a linear relationship for higher redshift. ##

    Parameters
    ----------
        MUV : `float'
        A value of the absolute magnitude in the UV (generally M1600) in the AB magnitude system.

    z : `float' 
        Redshift the attenuation is calculated at.

    Returns
    ------
    beta : `float'
        Value of the UV continuum paramaeter beta. 
    '''

    if (z >= 4.5 and z < 5.5): # z = 5 fits.
        if (MUV > -18.8):
            dB = -0.08
        else:
            dB = -0.17
        B = -2.05
        offset = 18.8
    elif (z >= 5.5 and z < 6.5): # z = 6 fits.
        if (MUV > -18.8):
            dB = -0.08
        else:
            dB = -0.24
        B = -2.22
        offset = 18.8

    elif (z >= 6.5 and z < 7.5): # z = 7 fits.
        dB = -0.20
        B = -2.05
        offset = 19.5
    elif (z >= 7.5 and z < 8.5): # z = 8 fits.
        dB = -0.15
        B = -2.13
        offset = 19.5
    elif (z >= 8.5 and z < 9.5): # z = 9 fits.
        dB = -0.16
        B = -2.19
        offset = 19.5
    elif (z >= 9.5 and z < 10.5): # z = 10 fits.
        dB = -0.16
        B = -2.16
        offset = 19.5

    beta = dB * (MUV + offset) + B

    return beta
        
def multiply(array):
    '''
    Performs element wise multiplication.

    Parameters
    ----------
    array : `~numpy.darray'
        The array to be multiplied.

    Returns
    -------
    total : `float'
        Total of the elements multiplied together.
    '''

    total = 1
    for i in range(0, len(array)):
        total *= array[i]
    return total

##

def Sum_Log(array):
    '''
    Performs an element wise sum of an array who's elements are in log-space.

    Parameters
    ----------
    array : array
    Array with elements in log-space.

    Returns
    ------
    sum_total : float
    Value of the elements taken to the power of 10 and summed.
    Units
    -----
    All units are kept the same as the inputs.
    '''

    sum_total = 0.0
    for i in range(0, len(array)):
        sum_total += 10**array[i]

    return sum_total

##

def Std_Log(array, mean):
    '''
    Calculates the standard deviation of an array with elements in log-space. 

    Parameters
    ----------
    array : array
    Array with elements in log-space.
    mean : float
    Mean of the array (not in log).

    Returns
    ------
    std : float
    Standard deviation of the input array taken to the power of 10. 
    Units
    -----
    All units are kept the same as the inputs.
    '''

    sum_total = 0.0
    for i in range(0, len(array)):
        sum_total += (10**array[i] - mean)**2

    sum_total *= 1.0/len(array)

    std = np.sqrt(sum_total)
    return std

###


def collect_across_tasks(mean_per_task, std_per_task, N_per_task, SnapList, 
                         BinSnapList=[], binned=False, m_bin_low=0.0, m_bin_high=0.0):
                
    """
    Reduces arrays that are unique to each task onto the master task.

    The dimensions of the input arrays will change slightly if we are collecting a statistics
    that is binned across e.g., halo mass or galaxy stellar mass.

    Parameters
    ----------

    mean_per_task, std_per_task, N_per_task: Nested 2D (or 3D if binned == True) arrays of floats.  
                                             Outer length is equal to the number of models.
                                             Inner length is equal to the number of snapshots the data has been calculated for.
                                             Most inner length is equal to the number of bins.
        Contains the mean/standard deviation/number of objects unique for each task.

    SnapList: Nested 2D arrays of integers.  Outer length is equal to the number of models.
        Contains the snapshot numbers the data has been calculated for each model. 

    BinSnapList: Nested 2D arrays of integers. Outer length is equal to the number of models.
        Often statistics are calculated for ALL snapshots but we only wish to plot for a subset of snapshots.
        This variable allows the binned data to be collected for only a subset of the snapshots.

    binned: Boolean.
        Dictates whether the collected data is a 2D or 3D array with the inner-most array being binned across e.g., halo mass.

    Returns
    ----------

    master_mean, master_std, master_N: Nested 2D (or 3D if binned == True) arrays of floats.
                                       Shape is identical to the input mean_per_task etc.
        If rank == 0 these contain the collected statistics.
        Otherwise these will be none.

    master_bin_middle: Array of floats.
        Contains the location of the middle of the bins for the data.         
    """


    master_mean = []
    master_std = []
    master_N = []

    master_bin_middle = []

    for model_number in range(0, len(SnapList)): 

        master_mean.append([])
        master_std.append([])
        master_N.append([])

        master_bin_middle.append([])

        # If we're collecting a binned statistic (e.g., binned across halo mass), then we need to perform the collecting per snapshot.
        if binned:
            count = 0 
            for snapshot_idx in range(len(SnapList[model_number])):
                if SnapList[model_number][snapshot_idx] == BinSnapList[model_number][count]:
                    master_mean[model_number], master_std[model_number], master_N[model_number] = calculate_pooled_stats(master_mean[model_number], master_std[model_number], master_N[model_number], mean_per_task[model_number][snapshot_idx], std_per_task[model_number][snapshot_idx], N_per_task[model_number][snapshot_idx])
                    master_bin_middle[model_number].append(np.arange(m_bin_low,
                                                                     m_bin_high+bin_width, 
                                                                     bin_width)[:-1] 
                                                           + bin_width * 0.5)

                    count += 1

                    if count == len(BinSnapList[model_number]):
                        break

        else:
            master_mean[model_number], master_std[model_number], master_N[model_number] = calculate_pooled_stats(master_mean[model_number], master_std[model_number], master_N[model_number], 
                                                                                                                 mean_per_task[model_number], std_per_task[model_number], 
                                                                                                                 N_per_task[model_number])

            if rank == 0:
                master_mean[model_number] = master_mean[model_number][0]
                master_std[model_number] = master_std[model_number][0]
                master_N[model_number] = master_N[model_number][0]
    return master_mean, master_std, master_N, master_bin_middle

###


def calculate_pooled_stats(mean_pool, std_pool, N_pool, mean_local, std_local, N_local):
    '''
    Calculates the pooled mean and standard deviation from multiple processors and appends it to an input array.
    Formulae taken from https://en.wikipedia.org/wiki/Pooled_variance
    As we only care about these stats on the rank 0 process, we make use of junk inputs/outputs for other ranks.

    NOTE: Since the input data may be an array (e.g. pooling the mean/std for a stellar mass function).

    Parameters
    ----------
    mean_pool, std_pool, N_pool : array of floats.
        Arrays that contain the current pooled means/standard deviation/number of data points (for rank 0) or just a junk input (for other ranks).
    mean_local, mean_std : float or array of floats.
        The non-pooled mean and standard deviation unique for each process.
    N_local : floating point number or array of floating point numbers. 
        Number of data points used to calculate the mean/standard deviation that is going to be added to the pool.
        NOTE: Use floating point here so we can use MPI.DOUBLE for all MPI functions.

    Returns
    -------
    mean_pool, std_pool : array of floats.
        Original array with the new pooled mean/standard deviation appended (for rank 0) or the new pooled mean/standard deviation only (for other ranks).

    Units
    -----
    All units are the same as the input.
    All inputs MUST BE real-space (not log-space).
    '''

    if isinstance(mean_local, list) == True:    
        if len(mean_local) != len(std_local):
            print("len(mean_local) = {0} \t len(std_local) = {1}".format(len(mean_local), len(std_local)))
            raise ValueError("Lengths of mean_local and std_local should be equal")
   
    if ((type(mean_local).__module__ == np.__name__) == True or (isinstance(mean_local, list) == True)): # Checks to see if we are dealing with arrays. 
    
        N_times_mean_local = np.multiply(N_local, mean_local)
        N_times_var_local = np.multiply(N_local, np.multiply(std_local, std_local))
        
        N_local = np.array(N_local).astype(float)
        N_times_mean_local = np.array(N_times_mean_local).astype(np.float32)

        if rank == 0: # Only rank 0 holds the final arrays so only it requires proper definitions.
            N_times_mean_pool = np.zeros_like(N_times_mean_local) 
            N_pool_function = np.zeros_like(N_local)
            N_times_var_pool = np.zeros_like(N_times_var_local)

            N_times_mean_pool = N_times_mean_pool.astype(np.float64) # Recast everything to double precision then use MPI.DOUBLE.
            N_pool_function = N_pool_function.astype(np.float64)
            N_times_var_pool = N_times_var_pool.astype(np.float64)
        else:
            N_times_mean_pool = None
            N_pool_function = None
            N_times_var_pool = None

        comm.Barrier()

        N_times_mean_local = N_times_mean_local.astype(np.float64)
        N_local = N_local.astype(np.float64)
        N_times_var_local = N_times_var_local.astype(np.float64)

        comm.Reduce([N_times_mean_local, MPI.DOUBLE], [N_times_mean_pool, MPI.DOUBLE], op = MPI.SUM, root = 0) # Sum the arrays across processors.
        comm.Reduce([N_local, MPI.DOUBLE],[N_pool_function, MPI.DOUBLE], op = MPI.SUM, root = 0)   
        comm.Reduce([N_times_var_local, MPI.DOUBLE], [N_times_var_pool, MPI.DOUBLE], op = MPI.SUM, root = 0)
        
    else:
    
        N_times_mean_local = N_local * mean_local
        N_times_var_local = N_local * std_local * std_local

        N_times_mean_pool = comm.reduce(N_times_mean_local, op = MPI.SUM, root = 0)
        N_pool_function = comm.reduce(N_local, op = MPI.SUM, root = 0)
        N_times_var_pool = comm.reduce(N_times_var_local, op = MPI.SUM, root = 0)
    
    if rank == 0:

        mean_pool_function = np.zeros((len(N_pool_function)))
        std_pool_function = np.zeros((len(N_pool_function)))

        for i in range(0, len(N_pool_function)):
            if N_pool_function[i] == 0:
                mean_pool_function[i] = 0.0
            else:
                mean_pool_function[i] = np.divide(N_times_mean_pool[i], N_pool_function[i])
            if N_pool_function[i] < 3:
                std_pool_function[i] = 0.0
            else:
                std_pool_function[i] = np.sqrt(np.divide(N_times_var_pool[i], N_pool_function[i]))
       
        mean_pool.append(mean_pool_function)
        std_pool.append(std_pool_function)
        N_pool.append(N_pool_function)

        return mean_pool, std_pool, N_pool
    else:
    
        return mean_pool, std_pool, N_pool_function # Junk return because non-rank 0 doesn't care.
##

def StellarMassFunction(SnapList, SMF, simulation_norm, FirstFile, LastFile, NumFile, ResolutionLimit_mean, model_tags, observations, paper_plot, output_tag):
    '''
    Calculates the stellar mass function for given galaxies with the option to overplot observations by Song et al. (2013) at z = 6, 7, 8 and/or Baldry et al. (2008) at z = 0.1. 
    Parallel compatible.
    NOTE: The plotting assumes the redshifts we are plotting at are (roughly) the same for each model. 

    Parameters
    ---------
    SnapList : Nested 'array-like`, SnapList[model_number0] = [snapshot0_model0, ..., snapshotN_model0], with length equal to the number of models.
        Snapshots that we plot the stellar mass function at for each model.
    SMF : Nested 2-dimensional array, SMF[model_number0][snapshot0]  = [bin0galaxies, ..., binNgalaxies], with length equal to the number of bins (NB_gal). 
        The count of galaxies within each stellar mass bin.  Bounds are given by 'm_gal_low' and 'm_gal_high' in bins given by 'bin_width'. 
    simulation_norm : array with length equal to the number of models.
        Denotes which simulation each model uses.  
        0 : MySim
        1 : Mini-Millennium
        2 : Tiamat (down to z = 5)
        3 : Extended Tiamat (down to z = 1.6ish).
        4 : Britton's Simulation
        5 : Kali
    FirstFile, LastFile, NumFile : array of integers with length equal to the number of models.
        The file numbers for each model that were read in (defined by the range between [FirstFile, LastFile] inclusive) and the TOTAL number of files for this model (we may only be plotting a subset of the volume). 
    ResolutionLimit_mean : array of floats with the same shape as SMF.
        This is the mean stellar mass for a halo with len (number of N-body simulation particles) between 'stellar_mass_halolen_lower' and 'stellar_mass_halolen_upper'. 
    model_tags : array of strings with length equal to the number of models.
        Strings that contain the tag for each model.  Will be placed on the plot.
    observations : int
        Denotes whether we want to overplot observational results. 
        0 : Don't plot anything. 
        1 : Plot Song et al. (2016) at z = 6, 7, 8. 
        2 : Plot Baldry et al. (2008) at z = 0.1.
        3 : Plot both of these.
    paper_plot : int
        Denotes whether we want to split the plotting over three panels (z = 6, 7, 8) for the paper or keep it all to one figure.
    output_tag : string
        Name of the file that will be generated. File will be saved in the current directory with the output format defined by the 'output_format' variable at the beggining of the file.

    Returns
    -------
    No returns.
    Generates and saves the plot (named via output_tag).

    Units
    -----
    Stellar Mass is in units of log10(Msun).
    '''

    ## Empty array initialization ##
    title = []
    normalization_array = []
    redshift_labels = []
    counts_array = []
    bin_middle_array = []

    for model_number in range(0, len(SnapList)):
        counts_array.append([])
        bin_middle_array.append([])
        redshift_labels.append([])

    ####
    for model_number in range(0, len(SnapList)): # Does this for each of the models. 

        ## Normalization for each model. ##
        if (simulation_norm[model_number] == 0):
            AllVars.Set_Params_Mysim()
        elif (simulation_norm[model_number] == 1):
            AllVars.Set_Params_MiniMill()
        elif (simulation_norm[model_number] == 2):
            AllVars.Set_Params_Tiamat()
        elif (simulation_norm[model_number] == 3):
            AllVars.Set_Params_Tiamat_extended()
        elif (simulation_norm[model_number] == 4):
            AllVars.Set_Params_Britton()       
        elif(simulation_norm[model_number] == 5):
            AllVars.Set_Params_Kali()
 
        box_factor = (LastFile[model_number] - FirstFile[model_number] + 1.0)/(NumFile[model_number]) # This factor allows us to take a sub-volume of the box and scale the results to represent the entire box.
        print("We are creating the stellar mass function using {0:.4f} of the box's volume.".format(box_factor))
        norm = pow(AllVars.BoxSize,3) / pow(AllVars.Hubble_h, 3) * bin_width * box_factor 
        normalization_array.append(norm)

        ####
    
        for snapshot_idx in range(0, len(SnapList[model_number])): # Loops for each snapshot in each model.
            tmp = 'z = %.2f' %(AllVars.SnapZ[SnapList[model_number][snapshot_idx]]) # Assigns a redshift label. 
            redshift_labels[model_number].append(tmp)

            ## We perform the plotting on Rank 0 so only this rank requires the final counts array. ##
            if rank == 0:
                counts_total = np.zeros_like(SMF[model_number][snapshot_idx])
            else:
                counts_total = None

            comm.Reduce([SMF[model_number][snapshot_idx], MPI.FLOAT], [counts_total, MPI.FLOAT], op = MPI.SUM, root = 0) # Sum all the stellar mass and pass to Rank 0.

            if rank == 0:
                counts_array[model_number].append(counts_total)
                bin_middle_array[model_number].append(np.arange(m_gal_low, m_gal_high+bin_width, bin_width)[:-1] + bin_width * 0.5)
            ####
    

    ## Plotting ##

    if rank == 0: # Plot only on rank 0.

        if paper_plot == 0:

            f = plt.figure()  
            ax = plt.subplot(111)  

            for model_number in range(0, len(SnapList)):
                for snapshot_idx in range(0, len(SnapList[model_number])):
                    if model_number == 0: # We assume the redshifts for each model are the same, we only want to put a legend label for each redshift once.
                        title = redshift_labels[model_number][snapshot_idx]
                    else:
                        title = ''
                    
                    plt.plot(bin_middle_array[model_number][snapshot_idx], counts_array[model_number][snapshot_idx] / normalization_array[model_number], color = PlotScripts.colors[snapshot_idx], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = title, linewidth = PlotScripts.global_linewidth) 

            #print(np.min(np.log10(ResolutionLimit_mean)))
        
            #ax.axvline(np.max(np.log10(ResolutionLimit_mean)), color = 'k', linewidth = PlotScripts.global_linewidth, linestyle = '--')    
            #ax.text(np.max(np.log10(ResolutionLimit_mean)) + 0.1, 1e-3, "Resolution Limit", color = 'k')
     
            for model_number in range(0, len(SnapList)): # Place legend labels for each of the models. NOTE: Placed after previous loop for proper formatting of labels. 
                plt.plot(1e100, 1e100, color = 'k', linestyle = PlotScripts.linestyles[model_number], label = model_tags[model_number], rasterized=True, linewidth = PlotScripts.global_linewidth)
        
            ## Adjusting axis labels/limits. ##

            plt.yscale('log', nonposy='clip')
            plt.axis([6, 11.5, 1e-6, 1e-0])

            ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}} \:[M_{\odot}]$', fontsize = PlotScripts.global_fontsize)
            ax.set_ylabel(r'$\Phi\ [\mathrm{Mpc}^{-3}\: \mathrm{dex}^{-1}]$', fontsize = PlotScripts.global_fontsize)
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
            ax.set_xticks(np.arange(6.0, 12.0))  

            if (observations == 1 or observations == 3): # If we wanted to plot Song.

                Obs.Get_Data_SMF()
                delta = 0.05
                caps = 5
                
                ## Song (2016) Plotting ##
                plt.errorbar(Obs.Song_SMF_z6[:,0], 10**Obs.Song_SMF_z6[:,1], yerr= (10**Obs.Song_SMF_z6[:,1] - 10**Obs.Song_SMF_z6[:,3], 10**Obs.Song_SMF_z6[:,2] - 10**Obs.Song_SMF_z6[:,1]), xerr = 0.25, capsize = caps, elinewidth = PlotScripts.global_errorwidth, alpha = 1.0, lw=2.0, marker='o', ls='none', label = 'Song 2015, z = 6', color = PlotScripts.colors[0], rasterized=True)
                plt.errorbar(Obs.Song_SMF_z7[:,0], 10**Obs.Song_SMF_z7[:,1], yerr= (10**Obs.Song_SMF_z7[:,1] - 10**Obs.Song_SMF_z7[:,3], 10**Obs.Song_SMF_z7[:,2] - 10**Obs.Song_SMF_z7[:,1]), xerr = 0.25, capsize = caps, alpha=0.75, elinewidth = PlotScripts.global_errorwidth, lw=1.0, marker='o', ls='none', label = 'Song 2015, z = 7', color = PlotScripts.colors[1], rasterized=True)
                plt.errorbar(Obs.Song_SMF_z8[:,0], 10**Obs.Song_SMF_z8[:,1], yerr= (10**Obs.Song_SMF_z8[:,1] - 10**Obs.Song_SMF_z8[:,3], 10**Obs.Song_SMF_z8[:,2] - 10**Obs.Song_SMF_z8[:,1]), xerr = 0.25, capsize = caps, alpha=0.75, elinewidth = PlotScripts.global_errorwidth, lw=1.0, marker='o', ls='none', label = 'Song 2015, z = 8', color = PlotScripts.colors[2], rasterized=True)
                ####

            if ((observations == 2 or observations == 3) and rank == 0): # If we wanted to plot Baldry.
                
                Baldry_xval = np.log10(10 ** Obs.Baldry_SMF_z0[:, 0]  /AllVars.Hubble_h/AllVars.Hubble_h)
                Baldry_xval = Baldry_xval - 0.26  # convert back to Chabrier IMF
                Baldry_yvalU = (Obs.Baldry_SMF_z0[:, 1]+Obs.Baldry_SMF_z0[:, 2]) * AllVars.Hubble_h*AllVars.Hubble_h*AllVars.Hubble_h
                Baldry_yvalL = (Obs.Baldry_SMF_z0[:, 1]-Obs.Baldry_SMF_z0[:, 2]) * AllVars.Hubble_h*AllVars.Hubble_h*AllVars.Hubble_h

                plt.fill_between(Baldry_xval, Baldry_yvalU, Baldry_yvalL, 
                    facecolor='purple', alpha=0.25, label='Baldry et al. 2008 (z=0.1)')
                ####

            leg = plt.legend(loc='lower left', numpoints=1, labelspacing=0.1)
            leg.draw_frame(False)  # Don't want a box frame
            for t in leg.get_texts():  # Reduce the size of the text
                t.set_fontsize(PlotScripts.global_legendsize)

            outputFile = './%s%s' %(output_tag, output_format) 
            plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
            print('Saved file to {0}'.format(outputFile))
            plt.close()

        if (paper_plot == 1):

            fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(16, 6))

            delta_fontsize = 0
            caps = 5
            ewidth = 1.5

            for model_number in range(0, len(SnapList)):
                for count in range(len(SnapList[model_number])):
                    w = np.where((counts_array[model_number][count] > 0))[0]
                    #label = r"$\mathbf{SAGE}$"
                    label = model_tags[model_number]
                    ax[count].plot(bin_middle_array[model_number][count][w], counts_array[model_number][count][w] 
                                   / normalization_array[model_number], color = PlotScripts.colors[model_number], 
                                   linestyle = PlotScripts.linestyles[model_number], rasterized = True, 
                                   label = label, linewidth = PlotScripts.global_linewidth)

                    tick_locs = np.arange(6.0, 12.0)
                    ax[count].set_xticklabels([r"$\mathbf{%d}$" % x for x in tick_locs], fontsize = PlotScripts.global_fontsize)
                    ax[count].set_xlim([6.8, 10.3])                    
                    ax[count].tick_params(which = 'both', direction='in', 
                                          width = PlotScripts.global_tickwidth)
                    ax[count].tick_params(which = 'major', length = PlotScripts.global_ticklength)
                    ax[count].tick_params(which = 'minor', length = PlotScripts.global_ticklength-2)
                    ax[count].set_xlabel(r'$\mathbf{log_{10} \: M_{*} \:[M_{\odot}]}$', 
                                         fontsize = PlotScripts.global_labelsize - delta_fontsize)
                    ax[count].xaxis.set_minor_locator(plt.MultipleLocator(0.25))
                    #ax[count].set_xticks(np.arange(6.0, 12.0))
                    
                    for axis in ['top','bottom','left','right']: # Adjust axis thickness.
                        ax[count].spines[axis].set_linewidth(PlotScripts.global_axiswidth)

            # Since y-axis is shared, only need to do this once.
            ax[0].set_yscale('log', nonposy='clip')
            ax[0].set_yticklabels([r"$\mathbf{10^{-5}}$",r"$\mathbf{10^{-5}}$",r"$\mathbf{10^{-4}}$", r"$\mathbf{10^{-3}}$",
                                   r"$\mathbf{10^{-2}}$",r"$\mathbf{10^{-1}}$"]) 
            ax[0].set_ylim([1e-5, 1e-1])
            #ax[0].set_ylabel(r'\mathbf{$\log_{10} \Phi\ [\mathrm{Mpc}^{-3}\: \mathrm{dex}^{-1}]}$', 
            ax[0].set_ylabel(r'$\mathbf{log_{10} \: \Phi\ [Mpc^{-3}\: dex^{-1}]}$', 
                             fontsize = PlotScripts.global_labelsize - delta_fontsize) 

            Obs.Get_Data_SMF()

            PlotScripts.Plot_SMF_z6(ax[0], errorwidth=ewidth, capsize=caps) 
            PlotScripts.Plot_SMF_z7(ax[1], errorwidth=ewidth, capsize=caps) 
            PlotScripts.Plot_SMF_z8(ax[2], errorwidth=ewidth, capsize=caps) 
            
            ####

            ax[0].text(0.7, 0.9, r"$\mathbf{z = 6}$", transform = ax[0].transAxes, fontsize = PlotScripts.global_fontsize - delta_fontsize)
            ax[1].text(0.7, 0.9, r"$\mathbf{z = 7}$", transform = ax[1].transAxes, fontsize = PlotScripts.global_fontsize - delta_fontsize)
            ax[2].text(0.7, 0.9, r"$\mathbf{z = 8}$", transform = ax[2].transAxes, fontsize = PlotScripts.global_fontsize - delta_fontsize)
                                       
            #leg = ax[0,0].legend(loc=2, bbox_to_anchor = (0.2, -0.5), numpoints=1, labelspacing=0.1)
            leg = ax[0].legend(loc='lower left', numpoints=1, labelspacing=0.1)
            leg.draw_frame(False)  # Don't want a box frame
            for t in leg.get_texts():  # Reduce the size of the text
                t.set_fontsize(PlotScripts.global_legendsize - 2)

            plt.tight_layout()
            #adjustprops = dict(left=0.15, bottom=0.1, right=0.97, top=0.93, wspace=0.05, hspace=0.05)
            #fig.subplots_adjust(**adjustprops)  
            outputFile = "{0}_paper{1}".format(output_tag, output_format) 
            plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
            print('Saved file to {0}'.format(outputFile))
            plt.close()


##

def plot_fesc(SnapList, mean_z_fesc, std_z_fesc, N_fesc, model_tags, output_tag):
    '''
    Plots the escape fraction as a function of redshift for the given galaxies. 
    Parallel compatible.
    Accepts 2D arrays of the escape fraction at each redshift for each model. 

    Parameters
    ---------
    SnapList : Nested array, SnapList[model_number0] = [snapshot0_model0, ..., snapshotN_model0], with length equal to the number of models.
        Snapshots for each model. 
    mean_z_fesc, std_z_fesc, N_fesc : Nested 2-dimensional array, mean_z_fesc[model_number0] = [z0_meanfesc, ..., zN_meanfesc], with length equal to the number of models 
        Mean/Standard deviation for fesc at each redshift. N_fesc is the number of data points in each bin. 
    model_tags : array of strings with length equal to the number of models.
        Strings that contain the tag for each model.  Will be placed on the plot.
    output_tag : string
        Name of the file that will be generated.

    Returns
    -------
    No returns.
    Generates and saves the plot (named via output_tag).   
    '''

    print("Plotting fesc as a function of redshift.")

    ## Array initialization ##
    pooled_mean_fesc = []
    pooled_std_fesc = []

    for model_number in range(0, len(SnapList)): # Loop for each model. 
    
        pooled_mean_fesc, pooled_std_fesc = calculate_pooled_stats(pooled_mean_fesc, pooled_std_fesc, mean_z_fesc[model_number], std_z_fesc[model_number], N_fesc[model_number]) # Calculates the pooled mean/standard deviation for this snapshot.  Only rank 0 receives a proper value here; the other ranks don't need this information. 
    
    if (rank == 0):
        ax1 = plt.subplot(111)

        for model_number in range(0, len(SnapList)):
    
            ## Calculate lookback time for each snapshot ##
            t = np.empty(len(SnapList[model_number]))
            for snapshot_idx in range(0, len(SnapList[model_number])):  
                t[snapshot_idx] = (t_BigBang - cosmo.lookback_time(AllVars.SnapZ[SnapList[model_number][snapshot_idx]]).value) * 1.0e3   
                    
            mean = pooled_mean_fesc[model_number]
            std = pooled_std_fesc[model_number]   

            ax1.plot(t, mean, color = PlotScripts.colors[model_number], linestyle = PlotScripts.linestyles[model_number], label = model_tags[model_number], linewidth = PlotScripts.global_linewidth)  
            ax1.fill_between(t, np.subtract(mean,std), np.add(mean,std), color = PlotScripts.colors[model_number], alpha = 0.25)

        ax1.xaxis.set_minor_locator(mtick.MultipleLocator(PlotScripts.time_tickinterval))
        ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.025))
        ax1.set_xlim(PlotScripts.time_xlim)

        ## Create a second axis at the top that contains the corresponding redshifts. ##
        ## The redshift defined in the variable 'z_plot' will be displayed. ##
        ax2 = ax1.twiny()

        t_plot = (t_BigBang - cosmo.lookback_time(PlotScripts.z_plot).value) * 1.0e3 # Corresponding time values on the bottom.
        z_labels = ["$%d$" % x for x in PlotScripts.z_plot] # Properly Latex-ize the labels.

        ax2.set_xlabel(r"$z$", size = PlotScripts.global_labelsize) 
        ax2.set_xlim(PlotScripts.time_xlim)
        ax2.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
        ax2.set_xticklabels(z_labels) # But label them as redshifts.

        ax1.set_ylim([0.0, 1.0])
        ax1.set_xlabel(r"$\mathrm{Time \: Since \: Big \: Bang \: [Myr]}$", size = PlotScripts.global_labelsize) 
        ax1.set_ylabel(r'$f_\mathrm{esc}$', fontsize = PlotScripts.global_fontsize) 

        leg = ax1.legend(loc=1, numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(PlotScripts.global_legendsize)

        plt.tight_layout()
        outputFile = './{0}{1}'.format(output_tag, output_format)
        plt.savefig(outputFile)  # Save the figure
        print('Saved file to {0}'.format(outputFile))
        plt.close()

##

def plot_fesc_galaxy(SnapList, PlotSnapList, simulation_norm, mean_galaxy_fesc, std_galaxy_fesc, N_galaxy_fesc, mean_halo_fesc, std_halo_fesc, N_halo_fesc, ResolutionLimit_mean, model_tags, output_tag):
    """
    Plots the escape fraction as a function of stellar/halo mass.
    Parallel compatible.
    Accepts 3D arrays of the escape fraction binned into Stellar Mass bins to plot the escape fraction for multiple models. 
    Mass units are 1e10 Msun (no h).

    Parameters
    ---------
    SnapList : Nested array, SnapList[model_number0] = [snapshot0_model0, ..., snapshotN_model0], with length equal to the number of models.
        Snapshots for each model. 
    simulation_norm : array with length equal to the number of models.
        Denotes which simulation each model uses.  
        0 : MySim
        1 : Mini-Millennium
        2 : Tiamat (down to z = 5)
        3 : Extended Tiamat (down to z = 1.6ish).
        4 : Britton's Simulation
        5 : Kali
    mean_galaxy_fesc, std_galaxy_fesc, N_galaxy_fesc : Nested 3-dimensional array, mean_galaxy_fesc[model_number0][snapshot0]  = [bin0_meanfesc, ..., binN_meanfesc], with length equal to the number of models. 
        Mean/Standard deviation for fesc in each stellar mass bin, for each [model_number] and [snapshot_number]. N_galaxy_fesc is the number of galaxies placed into each mass bin.
    mean_halo_fesc, std_halo_fesc, N_halo_fesc  Nested 3-dimensional array, mean_halo_fesc[model_number0][snapshot0]  = [bin0_meanfesc, ..., binN_meanfesc], with length equal to the number of models. 
        Identical to previous except using the halo virial mass for the binning rather than stellar mass. 
    ResolutionLimit_mean : array of floats with the same shape as mean_galaxy_fesc.
        This is the mean stellar mass for a halo with len (number of N-body simulation particles) between 'stellar_mass_halolen_lower' and 'stellar_mass_halolen_upper'. 
    model_tags : array of strings with length equal to the number of models.
        Strings that contain the tag for each model.  Will be placed on the plot.
    output_tag : string
        Name of the file that will be generated.

    Returns
    -------
    No returns.
    Generates and saves the plot (named via output_tag).  

    Units
    -----

    Mass units are 1e10 Msun (no h). 
    """

    print("Plotting fesc as a function of stellar mass.")

    ## Array initialization ##
    title = []
    redshift_labels = []

    mean_fesc_stellar_array = []
    std_fesc_stellar_array = []
    N_fesc_stellar_array = []

    mean_fesc_halo_array = []
    std_fesc_halo_array = []
    N_fesc_halo_array = []

    bin_middle_stellar_array = []
    bin_middle_halo_array = []

    for model_number in range(0, len(SnapList)):
        redshift_labels.append([])

        mean_fesc_stellar_array.append([])
        std_fesc_stellar_array.append([])
        N_fesc_stellar_array.append([])

        mean_fesc_halo_array.append([])
        std_fesc_halo_array.append([])
        N_fesc_halo_array.append([])

        bin_middle_stellar_array.append([])
        bin_middle_halo_array.append([])

    for model_number in range(0, len(SnapList)): 

        ## Normalization for each model. ##
        if (simulation_norm[model_number] == 0):
            AllVars.Set_Params_Mysim()
        elif (simulation_norm[model_number] == 1):
            AllVars.Set_Params_MiniMill()
        elif (simulation_norm[model_number] == 2):
            AllVars.Set_Params_Tiamat()
        elif (simulation_norm[model_number] == 3):
            AllVars.Set_Params_Tiamat_extended()
        elif (simulation_norm[model_number] == 4):
            AllVars.Set_Params_Britton()       
        elif(simulation_norm[model_number] == 5):
            AllVars.Set_Params_Kali()


        for snapshot_idx in range(0, len(SnapList[model_number])):
            tmp = 'z = %.2f' %(AllVars.SnapZ[SnapList[model_number][snapshot_idx]])
            redshift_labels[model_number].append(tmp)
 
            mean_fesc_stellar_array[model_number], std_fesc_stellar_array[model_number], N_fesc_stellar_array[model_number] = calculate_pooled_stats(mean_fesc_stellar_array[model_number], std_fesc_stellar_array[model_number], N_fesc_stellar_array[model_number], mean_galaxy_fesc[model_number][snapshot_idx], std_galaxy_fesc[model_number][snapshot_idx], N_galaxy_fesc[model_number][snapshot_idx]) 

            mean_fesc_halo_array[model_number], std_fesc_halo_array[model_number], N_fesc_halo_array[model_number] = calculate_pooled_stats(mean_fesc_halo_array[model_number], std_fesc_halo_array[model_number], N_fesc_halo_array[model_number], mean_halo_fesc[model_number][snapshot_idx], std_halo_fesc[model_number][snapshot_idx], N_halo_fesc[model_number][snapshot_idx]) 

            bin_middle_stellar_array[model_number].append(np.arange(m_gal_low, m_gal_high+bin_width, bin_width)[:-1] + bin_width * 0.5)
            bin_middle_halo_array[model_number].append(np.arange(m_low, m_high+bin_width, bin_width)[:-1] + bin_width * 0.5)
 
    if rank == 0:
        
        fig = plt.figure()  
        ax1 = fig.add_subplot(111)  
        ax2 = ax1.twinx()

        fig2 = plt.figure()
        ax3 = fig2.add_subplot(111)
        
       
        for model_number in range(0, len(SnapList)):

            print("There were a total of {0} galaxies over the entire redshift range.".format(sum(N_halo_fesc[model_number])))
            ## Normalization for each model. ##
            if (simulation_norm[model_number] == 0):
                AllVars.Set_Params_Mysim()
            elif (simulation_norm[model_number] == 1):
                AllVars.Set_Params_MiniMill()
            elif (simulation_norm[model_number] == 2):
                AllVars.Set_Params_Tiamat()
            elif (simulation_norm[model_number] == 3):
                AllVars.Set_Params_Tiamat_extended()
            elif (simulation_norm[model_number] == 4):
                AllVars.Set_Params_Britton()       
            elif(simulation_norm[model_number] == 5):
                AllVars.Set_Params_Kali()

            plot_count = 0
            for snapshot_idx in range(0, len(SnapList[model_number])):
                
                if (SnapList[model_number][snapshot_idx] == PlotSnapList[model_number][plot_count]):

                    if (model_number == 0):
                        label = redshift_labels[model_number][snapshot_idx]
                    else:
                        label = ""

                    ## Plots as a function of stellar mass ##
                    w = np.where((N_galaxy_fesc[model_number][snapshot_idx] < 1))[0] # If there are no galaxies in the bin we don't want to plot. 
                    N_galaxy_fesc[model_number][snapshot_idx][w] = np.nan 
                    mean_fesc_stellar_array[model_number][snapshot_idx][w] = np.nan

                    ax1.plot(bin_middle_stellar_array[model_number][snapshot_idx], mean_fesc_stellar_array[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = label, linewidth = PlotScripts.global_linewidth) # Plots the escape fraction on the left.
    #                ax2.plot(bin_middle_stellar_array[model_number][snapshot_idx], N_galaxy_fesc[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = '-.', rasterized = True, linewidth = PlotScripts.global_linewidth) # And the number of galaxies in the bin on the right.

                    print("Resolution limit for model {0} at snapshot {1} is {2}".format(model_number, snapshot_idx, np.log10(ResolutionLimit_mean[model_number][snapshot_idx])))
                    if plot_count == 2 and model_number == 0:
                        ax1.axvline(np.log10(ResolutionLimit_mean[model_number][snapshot_idx]), color = 'k', linewidth = PlotScripts.global_linewidth, linestyle = '--')    

                    ## Plots as a function of halo mass ##
                    w = np.where((N_halo_fesc[model_number][snapshot_idx] < 1))[0] # If there are no galaxies in the bin we don't want to plot.                     
                    N_halo_fesc[model_number][snapshot_idx][w] = np.nan 
                    mean_fesc_halo_array[model_number][snapshot_idx][w] = np.nan

                    '''
                    if (model_number == 0):
                        print(bin_middle_halo_array[model_number][snapshot_idx])
                        print(mean_fesc_halo_array[model_number][snapshot_idx])
                    '''
                    ax3.plot(bin_middle_halo_array[model_number][snapshot_idx], mean_fesc_halo_array[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = label, linewidth = PlotScripts.global_linewidth) # Plots the escape fraction on the left.
    #                ax4.plot(bin_middle_halo_array[model_number][snapshot_idx], N_halo_fesc[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = '-.', rasterized = True, linewidth = PlotScripts.global_linewidth) # And the number of halos in the bin on the right.

                    plot_count += 1                
                    if (plot_count == len(PlotSnapList[model_number])):
                        break

        for model_number in range(0, len(SnapList)): # Just plot some garbage to get the legend labels correct.
            ax1.plot(np.nan, np.nan, color = 'k', linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = model_tags[model_number], linewidth = PlotScripts.global_linewidth)
            ax3.plot(np.nan, np.nan, color = 'k', linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = model_tags[model_number], linewidth = PlotScripts.global_linewidth)

        ## Stellar Mass plots ##
 
        ax1.axhline(0.2, 0, 100, color ='k', linewidth = PlotScripts.global_linewidth, linestyle = '-.')

        ax1.set_xlabel(r'$\log_{10}\ M_*\ [M_{\odot}]$', size = PlotScripts.global_fontsize) 
        ax1.set_ylabel(r'$f_\mathrm{esc}$', size = PlotScripts.global_fontsize)
        ax2.set_ylabel(r'$N_\mathrm{gal}$', size = PlotScripts.global_fontsize)
        ax1.set_xlim([4.0, 12])
        ax1.set_ylim([-0.05, 1.0])   

        ax1.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
        ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.025))
        
        ax2.set_yscale('log', nonposy='clip')

        leg = ax1.legend(loc=9, numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize('medium')

        ## Halo mass plots ##

        ax3.axhline(0.35, 0, 100, color ='k', linewidth = PlotScripts.global_linewidth, linestyle = '-.')
        ax3.axvline(np.log10(32.0*AllVars.PartMass / AllVars.Hubble_h), color = 'k', linewidth = PlotScripts.global_linewidth, linestyle = '-.')   
        ax3.text(10.7, 0.26, r"$f_\mathrm{esc} = 0.35$", color = 'k', size = PlotScripts.global_fontsize)
 
        ax3.set_xlabel(r'$\log_{10}\ M_\mathrm{vir}\ [M_{\odot}]$', size = PlotScripts.global_fontsize) 
        ax3.set_ylabel(r'$f_\mathrm{esc}$', size = PlotScripts.global_fontsize)
        #ax4.set_ylabel(r'$N_\mathrm{Halo}$', size = PlotScripts.global_fontsize)
        ax3.set_xlim([8.6, 11.75])
        ax3.set_ylim([-0.05, 1.0])   

        ax3.set_xticks(np.arange(9.0, 11.0))  
        ax3.xaxis.set_minor_locator(mtick.MultipleLocator(0.25))
        ax3.yaxis.set_minor_locator(mtick.MultipleLocator(0.05))
        
        #ax4.set_yscale('log', nonposy='clip')

        leg = ax3.legend(loc='upper left', bbox_to_anchor=(0.3, 1.02), numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(PlotScripts.global_legendsize)
        ## Output ##

        outputFile = './%s%s' %(output_tag, output_format)
        fig.savefig(outputFile, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile))

        outputFile = './%s_Halo%s' %(output_tag, output_format)
        fig2.savefig(outputFile, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile))

        plt.close(fig)
        plt.close(fig2)

##

def plot_ejectedfraction(SnapList, mean_mvir_ejected, std_mvir_ejected, N_ejected, model_tags, output_tag): 
    '''
    Plots the ejected fraction as a function of the halo mass. 
    Parallel compatible.
    Accepts a 3D array of the ejected fraction so we can plot for multiple models and redshifts. 

    Parameters
    ---------
    SnapList : Nested array, SnapList[model_number0] = [snapshot0_model0, ..., snapshotN_model0], with length equal to the number of models.
    Snapshots for each model. 
    mean_mvir_ejected, std_mvir_ejected, N_ejected : Nested 3-dimensional array, mean_mvir_ejected[model_number0][snapshot0]  = [bin0_meanejected, ..., binN_meanejected], with length equal to the number of models. 
    Mean/Standard deviation for the escape fraction binned into Halo Mass bins. N_ejected is the number of data points in each bin. Bounds are given by 'm_low' and 'm_high' in bins given by 'bin_width'.   
    model_tags : array of strings with length equal to the number of models.
    Strings that contain the tag for each model.  Will be placed on the plot.
    output_tag : string
    Name of the file that will be generated.

    Returns
    -------
    No returns.
    Generates and saves the plot (named via output_tag).

    Units
    -----
    Halo Mass is in units of log10(Msun). 
    '''

    print("Plotting the Ejected Fraction as a function of halo mass.")

    ## Array initialization. ##
    title = []
    redshift_labels = []

    mean_ejected_array = []
    std_ejected_array = []

    mean_halomass_array = []
    std_halomass_array = []

    bin_middle_array = []

    for model_number in range(0, len(SnapList)):
        redshift_labels.append([])

        mean_ejected_array.append([])
        std_ejected_array.append([])

        mean_halomass_array.append([])
        std_halomass_array.append([])

        bin_middle_array.append([])
    
    bin_width = 0.1
 
    for model_number in range(0, len(SnapList)): 
        for snapshot_idx in range(0, len(SnapList[model_number])):
            print("Doing Snapshot {0}".format(SnapList[model_number][snapshot_idx]))
            tmp = 'z = %.2f' %(AllVars.SnapZ[SnapList[model_number][snapshot_idx]])
            redshift_labels[model_number].append(tmp)
            
            mean_ejected_array[model_number], std_ejected_array[model_number] = calculate_pooled_stats(mean_ejected_array[model_number], std_ejected_array[model_number], mean_mvir_ejected[model_number][snapshot_idx], std_mvir_ejected[model_number][snapshot_idx], N_ejected[model_number][snapshot_idx]) # Calculates the pooled mean/standard deviation for this snapshot.  Only rank 0 receives a proper value here; the other ranks don't need this information. 

            bin_middle_array[model_number].append(np.arange(m_low, m_high+bin_width, bin_width)[:-1] + bin_width * 0.5)
    
    if rank == 0:
        f = plt.figure()  
        ax1 = plt.subplot(111)  

        for model_number in range(0, len(SnapList)):
            for snapshot_idx in range(0, len(SnapList[model_number])): 
                ax1.plot(bin_middle_array[model_number][snapshot_idx], mean_ejected_array[model_number][snapshot_idx], color = PlotScripts.colors[snapshot_idx], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = redshift_labels[model_number][snapshot_idx], linewidth = PlotScripts.global_linewidth) 
                

        for model_number in range(0, len(SnapList)): # Just plot some garbage to get the legend labels correct.
            ax1.plot(np.nan, np.nan, color = 'k', linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = model_tags[model_number], linewidth = PlotScripts.global_linewidth)

        ax1.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$', size = PlotScripts.global_fontsize) 
        ax1.set_ylabel(r'$\mathrm{Ejected \: Fraction}$', size = PlotScripts.global_fontsize)
        ax1.set_xlim([8.0, 12])
        ax1.set_ylim([-0.05, 1.0])   

        ax1.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
        ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.025))
        
        leg = ax1.legend(loc=1, numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize('medium')

        outputFile = './%s%s' %(output_tag, output_format)
        plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile))
        plt.close()

##


def plot_mvir_fesc(SnapList, mass_central, fesc, model_tags, output_tag): 

    title = []
    redshift_labels = []

    mean_fesc_array = []
    std_fesc_array = []

    mean_halomass_array = []
    std_halomass_array = []

    bin_middle_array = []

    for model_number in range(0, len(SnapList)):
        redshift_labels.append([])

    mean_fesc_array.append([])
    std_fesc_array.append([])

    mean_halomass_array.append([])
    std_halomass_array.append([])

    bin_middle_array.append([])
    print("Plotting fesc against Mvir") 
    
    binwidth = 0.1
    Frequency = 1  
 
    for model_number in range(0, len(SnapList)): 
        for snapshot_idx in range(0, len(SnapList[model_number])):
            print("Doing Snapshot {0}".format(SnapList[model_number][snapshot_idx]))
            tmp = 'z = %.2f' %(AllVars.SnapZ[SnapList[model_number][snapshot_idx]])
            redshift_labels[model_number].append(tmp)

            minimum_mass = np.floor(min(mass_central[model_number][snapshot_idx])) - 10*binwidth
            maximum_mass = np.floor(max(mass_central[model_number][snapshot_idx])) + 10*binwidth

            minimum_mass = 6.0
            maximum_mass = 12.0

            binning_minimum = comm.allreduce(minimum_mass, op = MPI.MIN)
            binning_maximum = comm.allreduce(maximum_mass, op = MPI.MAX)
            
            halomass_nonlog = [10**x for x in mass_central[model_number][snapshot_idx]]
            (mean_fesc, std_fesc, N, bin_middle) = AllVars.Calculate_2D_Mean(mass_central[model_number][snapshot_idx], fesc[model_number][snapshot_idx], binwidth, binning_minimum, binning_maximum)

            mean_fesc_array[model_number], std_fesc_array[model_number] = calculate_pooled_stats(mean_fesc_array[model_number], std_fesc_array[model_number], mean_fesc, std_fesc, N)
            mean_halomass_array[model_number], std_halomass_array[model_number] = calculate_pooled_stats(mean_halomass_array[model_number], std_halomass_array[model_number], np.mean(halomass_nonlog), np.std(halomass_nonlog), len(mass_central[model_number][snapshot_idx]))

            ## If want to do mean/etc of halo mass need to update script. ##
            bin_middle_array[model_number].append(bin_middle)
        
        mean_halomass_array[model_number] = np.log10(mean_halomass_array[model_number]) 
        
    if rank == 0:
        f = plt.figure()  
        ax1 = plt.subplot(111)  

        for model_number in range(0, len(SnapList)):
            for snapshot_idx in range(0, len(SnapList[model_number])):
                if model_number == 0:
                    title = redshift_labels[model_number][snapshot_idx]
                else:
                    title = ''
                
                mean = mean_fesc_array[model_number][snapshot_idx]
                std = std_fesc_array[model_number][snapshot_idx] 
                bin_middle = bin_middle_array[model_number][snapshot_idx]

                ax1.plot(bin_middle, mean, color = colors[snapshot_idx], linestyle = linestyles[model_number], rasterized = True, label = title)
                #ax1.scatter(mean_halomass_array[model_number][snapshot_idx], np.mean(~np.isnan(mean)), color = colors[snapshot_idx], marker = 'o', rasterized = True, s = 40, lw = 3)  
                if (len(SnapList) == 1):
                        ax1.fill_between(bin_middle, np.subtract(mean,std), np.add(mean,std), color = colors[snapshot_idx], alpha = 0.25)

        ax1.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$', size = PlotScripts.global_fontsize) 
        ax1.set_ylabel(r'$f_\mathrm{esc}$', size = PlotScripts.global_fontsize) 
        #ax1.set_xlim([8.5, 12])
        #ax1.set_ylim([0.0, 1.0])   

        ax1.xaxis.set_minor_locator(mtick.MultipleLocator(0.1))
#       ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.1))
    
#       ax1.set_yscale('log', nonposy='clip')
#       for model_number in range(0, len(SnapList)):
#       ax1.plot(1e100, 1e100, color = 'k', ls = linestyles[model_number], label = model_tags[model_number], rasterized=True)
    
    
        leg = ax1.legend(loc='upper left', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize('medium')

        outputFile = './' + output_tag + output_format
        plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
        print('Saved file to'.format(outputFile))
        plt.close()
##

def plot_mvir_Ngamma(SnapList, mean_mvir_Ngamma, std_mvir_Ngamma, N_Ngamma, model_tags, output_tag,fesc_prescription=None, fesc_normalization=None, fitpath=None): 
    '''
    Plots the number of ionizing photons (pure ngamma times fesc) as a function of halo mass. 
    Parallel compatible.
    The input data has been binned as a function of halo virial mass (Mvir), with the bins defined at the top of the file (m_low, m_high, bin_width). 
    Accepts 3D arrays to plot ngamma for multiple models. 

    Parameters
    ----------
    SnapList : Nested array, SnapList[model_number0] = [snapshot0_model0, ..., snapshotN_model0], with length equal to the number of models.
    Snapshots for each model. 
    mean_mvir_Ngamma, std_mvir_Ngamma, N_Ngamma : Nested 2-dimensional array, mean_mvir_Ngamma[model_number0][snapshot0]  = [bin0_meanNgamma, ..., binN_meanNgamma], with length equal to the number of bins. 
    Mean/Standard deviation/number of data points in each halo mass (Mvir) bin.
    The number of photons is in units of 1.0e50 s^-1.   
    model_tags : array of strings with length equal to the number of models.
    Strings that contain the tag for each model.  Will be placed on the plot.
    output_tag : string
    Name of the file that will be generated.
    fesc_prescription : int (optional)
    If this parameter is defined, we will save the Mvir-Ngamma results in a text file (not needed if not saving).
    Number that controls what escape fraction prescription was used to generate the escape fractions.
    0 : Constant, fesc = Constant.
    1 : Scaling with Halo Mass, fesc = A*Mh^B.
    2 : Scaling with ejected fraction, fesc = fej*A + B.
    fesc_normalization : float (if fesc_prescription == 0) or `numpy.darray' with length 2 (if fesc_prescription == 1 or == 2) (optional).
    If this parameter is defined, we will save the Mvir-Ngamma results in a text file (not needed if not saving).
    Parameter not needed if you're not saving the Mvir-Ngamma results.
    If fesc_prescription == 0, gives the constant value for the escape fraction.
    If fesc_prescription == 1 or == 2, gives A and B with the form [A, B].
    fitpath : string (optional) 
    If this parameter is defined, we will save the Mvir-Ngamma results in a text file (not needed if not saving).
    Defines the base path for where we are saving the results.

    Returns
    -------
    No returns.
    Generates and saves the plot (named via output_tag).

    Units
    -----
    Ngamma is in units of 1.0e50 s^-1. 
    '''

    print("Plotting ngamma*fesc against the halo mass") 

    ## Array initialization. ##
    title = []
    redshift_labels = []

    mean_ngammafesc_array = []
    std_ngammafesc_array = []

    mean_halomass_array = []
    std_halomass_array = []

    bin_middle_array = []

    for model_number in range(0, len(SnapList)):
        redshift_labels.append([])

    mean_ngammafesc_array.append([])
    std_ngammafesc_array.append([])

    mean_halomass_array.append([])
    std_halomass_array.append([])

    bin_middle_array.append([])
    
    for model_number in range(0, len(SnapList)): 
        for snapshot_idx in range(0, len(SnapList[model_number])):
            print("Doing Snapshot {0}".format(SnapList[model_number][snapshot_idx]))
            tmp = 'z = %.2f' %(AllVars.SnapZ[SnapList[model_number][snapshot_idx]])
            redshift_labels[model_number].append(tmp)

            N = N_Ngamma[model_number][snapshot_idx]
            
            mean_ngammafesc_array[model_number], std_ngammafesc_array[model_number] = calculate_pooled_stats(mean_ngammafesc_array[model_number], std_ngammafesc_array[model_number], mean_mvir_Ngamma[model_number][snapshot_idx], std_mvir_Ngamma[model_number][snapshot_idx], N) # Collate the values from all processors.   
            bin_middle_array[model_number].append(np.arange(m_low, m_high+bin_width, bin_width)[:-1] + bin_width * 0.5) 
    
    if rank == 0:
        f = plt.figure()  
        ax1 = plt.subplot(111)  

        for model_number in range(0, len(SnapList)):
            count = 0
            for snapshot_idx in range(0, len(SnapList[model_number])):
                if model_number == 0:
                    title = redshift_labels[model_number][snapshot_idx]
                else:
                    title = ''

                mean = np.zeros((len(mean_ngammafesc_array[model_number][snapshot_idx])), dtype = np.float32)
                std = np.zeros((len(mean_ngammafesc_array[model_number][snapshot_idx])), dtype=np.float32)

                for i in range(0, len(mean)):
                    if(mean_ngammafesc_array[model_number][snapshot_idx][i] < 1e-10):
                        mean[i] = np.nan
                        std[i] = np.nan
                    else:   
                        mean[i] = np.log10(mean_ngammafesc_array[model_number][snapshot_idx][i] * 1.0e50) # Remember that the input data is in units of 1.0e50 s^-1.
                        std[i] = 0.434 * std_ngammafesc_array[model_number][snapshot_idx][i] / mean_ngammafesc_array[model_number][snapshot_idx][i] # We're plotting in log space so the standard deviation is 0.434*log10(std)/log10(mean).

                bin_middle = bin_middle_array[model_number][snapshot_idx]
    
                if (count < 4): # Only plot at most 5 lines.
                    ax1.plot(bin_middle, mean, color = PlotScripts.colors[snapshot_idx], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = title, linewidth = PlotScripts.global_linewidth)  
                    count += 1
                ## In this block we save the Mvir-Ngamma results to a file. ##
                if (fesc_prescription == None or fesc_normalization == None or fitpath == None):
                    raise ValueError("You've specified you want to save the Mvir-Ngamma results but haven't provided an escape fraction prescription, normalization and base path name")
                # Note: All the checks that escape fraction normalization was written correctly were performed in 'calculate_fesc()', hence it will be correct by this point and we don't need to double check. 

                if (fesc_prescription[model_number] == 0): # Slightly different naming scheme for the constant case (it only has a float for fesc_normalization).
                    fname = "%s/fesc%d_%.3f_z%.3f.txt" %(fitpath, fesc_prescription[model_number], fesc_normalization[model_number], AllVars.SnapZ[SnapList[model_number][snapshot_idx]])
                elif (fesc_prescription[model_number] == 1 or fesc_prescription[model_number] == 2):    
                    fname = "%s/fesc%d_A%.3eB%.3f_z%.3f.txt" %(fitpath, fesc_prescription[model_number], fesc_normalization[model_number][0], fesc_normalization[model_number][1], AllVars.SnapZ[SnapList[model_number][snapshot_idx]])
                f = open(fname, "w+")
                if not os.access(fname, os.W_OK):
                    print("The filename is {0}".format(fname))
                    raise ValueError("Can't write to this file.")
        
                for i in range(0, len(bin_middle)):
                    f.write("%.4f %.4f %.4f %d\n" %(bin_middle[i], mean[i], std[i], N_Ngamma[model_number][snapshot_idx][i]))
                f.close() 
                print("Wrote successfully to file {0}".format(fname))
            ##

        for model_number in range(0, len(SnapList)): # Just plot some garbage to get the legend labels correct.
            ax1.plot(np.nan, np.nan, color = 'k', linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = model_tags[model_number], linewidth = PlotScripts.global_linewidth)
    
        ax1.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$', size = PlotScripts.global_fontsize) 
        ax1.set_ylabel(r'$\log_{10}\ \dot{N}_\gamma \: f_\mathrm{esc} \: [\mathrm{s}^{-1}]$', size = PlotScripts.global_fontsize) 
        ax1.set_xlim([8.5, 12])

        ax1.xaxis.set_minor_locator(mtick.MultipleLocator(0.1)) 
    
        leg = ax1.legend(loc='upper left', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize('medium')

        outputFile = './' + output_tag + output_format
        plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
        print('Saved file to'.format(outputFile))
    
        plt.close()


def bin_Simfast_halos(RedshiftList, SnapList, halopath, fitpath, fesc_prescription, fesc_normalization, GridSize, output_tag):
   
    for model_number in range(0, len(fesc_prescription)):
        for halo_z_idx in range(0, len(RedshiftList)):
            snapshot_idx = min(range(len(SnapList)), key=lambda i: abs(SnapList[i]-RedshiftList[halo_z_idx])) # This finds the index of the simulation redshift that most closely matches the Halo redshift.
            print("Binning Halo redshift {0}".format(RedshiftList[halo_z_idx]))
            print("For the Halo redshift {0:.3f} the nearest simulation redshift is {1:.3f}".format(RedshiftList[halo_z_idx], SnapList[snapshot_idx])) 
            if (fesc_prescription[model_number] == 0):
                fname = "%s/fesc%d_%.3f_z%.3f.txt" %(fitpath, fesc_prescription[model_number], fesc_normalization[model_number], AllVars.SnapZ[snapshot_idx]) 
            elif (fesc_prescription[model_number] == 1 or fesc_prescription[model_number] == 2):
                fname = "%s/fesc%d_A%.3eB%.3f_z%.3f.txt" %(fitpath, fesc_prescription[model_number], fesc_normalization[model_number][0], fesc_normalization[model_number][1], AllVars.SnapZ[snapshot_idx])

            print("Reading in file {0}".format(fname))
            ## Here we read in the results from the Mvir-Ngamma binning. ##
            f = open(fname, 'r')
            fit_mvir, fit_mean, fit_std, fit_N = np.loadtxt(f, unpack = True)
            f.close()

            ## Here we read in the halos created by Simfast21 ##
            # The data file has the structure:
                # long int N_halos
                # Then an entry for each halo:
                    # float Mass
                    # float x, y, z positions.
                # NOTE: The x,y,z positions are the grid indices but are still floats (because Simfast21 is weird like that).

            Halodesc_full = [ 
                 ('Halo_Mass', np.float32),
                 ('Halo_x', np.float32),
                 ('Halo_y', np.float32),
                 ('Halo_z', np.float32)
                ]

            names = [Halodesc_full[i][0] for i in range(len(Halodesc_full))]
            formats = [Halodesc_full[i][1] for i in range(len(Halodesc_full))]
            Halo_Desc = np.dtype({'names':names, 'formats':formats}, align=True)

            fname = "%s/halonl_z%.3f_N%d_L100.0.dat.catalog" %(halopath, RedshiftList[halo_z_idx], GridSize)
            f = open(fname, 'rb')
            N_Halos = np.fromfile(f, count = 1, dtype = np.long)    
            Halos = np.fromfile(f, count = N_Halos, dtype = Halo_Desc)  

            binned_nion = np.zeros((GridSize*GridSize*GridSize), dtype = float32) # This grid will contain the ionizing photons that results from the binning.
            binned_Halo_Mass = np.digitize(np.log10(Halos['Halo_Mass']), fit_mvir) # Places the Simfast21 halos into the correct halo mass bins defined by the Mvir-Ngamma results.
            binned_Halo_Mass[binned_Halo_Mass == len(fit_mvir)] = len(fit_mvir) - 1 # Fixes up the edge case.

            ## Fore each Halo we now assign it an ionizing flux. ##
            # This flux is determined by drawing a random number from a normal distribution with mean and standard deviation given by the Mvir-Ngamma results.
            # NOTE: Remember the Mvir-Ngamma results are in units of log10(s^-1).
            fit_nan = 0
            for i in range(0, N_Halos):
                if(np.isnan(fit_mean[binned_Halo_Mass[i]]) == True or np.isnan(fit_std[binned_Halo_Mass[i]]) == True): # This halo had mass that was not covered by the Mvir-Ngamma fits.
                    fit_nan += 1
                    continue
                nion_halo = np.random.normal(fit_mean[binned_Halo_Mass[i]], fit_std[binned_Halo_Mass[i]])

                ## Because of how Simfast21 does their binning, we have some cases where the Halos are technically outside the box. Just fix them up. ##
                x_grid = int(Halos['Halo_x'][i])
                if x_grid >= GridSize:
                    x_grid = GridSize - 1
                if x_grid < 0:
                    x_grid = 0

                y_grid = int(Halos['Halo_y'][i])
                if y_grid >= GridSize:
                    y_grid = GridSize - 1
                if y_grid < 0:
                    y_grid = 0
                    
                z_grid = int(Halos['Halo_z'][i])
                if z_grid >= GridSize:
                    z_grid = GridSize - 1
                if z_grid < 0:
                    z_grid = 0
                
                idx = x_grid * GridSize*GridSize + y_grid * GridSize + z_grid
                binned_nion[idx]  += pow(10, nion_halo)/1.0e50 
#            print"We had %d halos (out of %d, so %.4f fraction) that had halo mass that was not covered by the Mvir-Ngamma results." %(fit_nan, N_Halos, float(fit_nan)/float(N_Halos))
 #           print "There were %d cells with a non-zero ionizing flux." %(len(binned_nion[binned_nion != 0]))

            binned_nion = binned_nion.reshape((GridSize,GridSize,GridSize))
            cut_slice = 0
            cut_width = 512
            nion_slice = binned_nion[:,:, cut_slice:cut_slice+cut_width].mean(axis=-1)*1.0e50

            ax1 = plt.subplot(211)
            
            im = ax1.imshow(np.log10(nion_slice), interpolation='bilinear', origin='low', extent =[0,AllVars.BoxSize,0,AllVars.BoxSize], cmap = 'Purples', vmin = 48, vmax = 53)

            cbar = plt.colorbar(im, ax = ax1)
            cbar.set_label(r'$\mathrm{log}_{10}N_{\gamma} [\mathrm{s}^{-1}]$')

            ax1.set_xlabel(r'$\mathrm{x}  (h^{-1}Mpc)$')
            ax1.set_ylabel(r'$\mathrm{y}  (h^{-1}Mpc)$')

            ax1.set_xlim([0.0, AllVars.BoxSize])
            ax1.set_ylim([0.0, AllVars.BoxSize])

            title = r"$z = %.3f$" %(RedshiftList[halo_z_idx])
            ax1.set_title(title)
    
            ax2 = plt.subplot(212)

            w = np.where((Halos['Halo_z'][:] > cut_slice) & (Halos['Halo_z'][:] <= cut_slice + cut_width))[0]

            x_plot = Halos['Halo_x'] * float(AllVars.BoxSize)/float(GridSize)
            y_plot = Halos['Halo_y'] * float(AllVars.BoxSize)/float(GridSize)
            z_plot = Halos['Halo_z'][w] * float(AllVars.BoxSize)/float(GridSize)
            
            ax2.scatter(x_plot[w], y_plot[w], s = 2, alpha = 0.5)

            ax2.set_xlabel(r'$\mathrm{x}  (h^{-1}Mpc)$')
            ax2.set_ylabel(r'$\mathrm{y}  (h^{-1}Mpc)$')

            ax2.set_xlim([0.0, AllVars.BoxSize])
            ax2.set_ylim([0.0, AllVars.BoxSize])

            tmp = "z%.3f" %(RedshiftList[halo_z_idx])
            
            plt.tight_layout()
            outputFile = './' + output_tag + tmp + output_format
            plt.savefig(outputFile)  # Save the figure
            print('Saved file to {0}'.format(outputFile))
            plt.close()
            
def plot_photoncount(SnapList, sum_nion, simulation_norm, FirstFile, LastFile, NumFiles, model_tags, output_tag): 
    '''
    Plots the ionizing emissivity as a function of redshift. 
    We normalize the emissivity to Mpc^-3 and this function allows the read-in of only a subset of the volume.
    Parallel compatible.

    Parameters
    ---------
    SnapList : Nested array, SnapList[model_number0] = [snapshot0_model0, ..., snapshotN_model0], with length equal to the number of models.
        Snapshots for each model, defines the x-axis we plot against.
    sum_nion : Nested 1-dimensional array, sum_nion[z0, z1, ..., zn], with length equal to the number of redshifts. 
        Number of escape ionizing photons (i.e., photon rate times the local escape fraction) at each redshift.
        In units of 1.0e50 s^-1.
    simulation_norm : array  of ints with length equal to the number of models.
        Denotes which simulation each model uses.  
        0 : MySim
        1 : Mini-Millennium
        2 : Tiamat (down to z = 5)
        3 : Extended Tiamat (down to z = 1.6ish).
        4 : Britton's Simulation
    FirstFile, LastFile, NumFile : array of integers with length equal to the number of models.
        The file numbers for each model that were read in (defined by the range between [FirstFile, LastFile] inclusive) and the TOTAL number of files for this model (we may only be plotting a subset of the volume). 
    model_tags : array of strings with length equal to the number of models.
        Strings that contain the tag for each model.  Will be placed on the plot.
    output_tag : string
        Name of the file that will be generated.

    Returns
    -------
    No returns.
    Generates and saves the plot (named via output_tag).

    Units
    -----
    sum_nion is in units of 1.0e50 s^-1.
    '''
    print("Plotting the ionizing emissivity.")

    sum_array = []

    for model_number in range(0, len(SnapList)):
        if(simulation_norm[model_number] == 0):
            AllVars.Set_Params_Mysim()
        if(simulation_norm[model_number] == 1):
            AllVars.Set_Params_MiniMill()
        elif(simulation_norm[model_number] == 3):
            AllVars.Set_Params_Tiamat_extended()
        elif(simulation_norm[model_number] == 4):
            AllVars.Set_Params_Britton()
        elif(simulation_norm[model_number] == 5):
            AllVars.Set_Params_Kali()
        else: 
            print("Simulation norm was set to {0}.".format(simulation_norm[model_number]))
            raise ValueError("This option has been implemented yet.  Get your head in the game Jacob!")
 
        sum_array.append([])
        for snapshot_idx in range(0, len(SnapList[model_number])):
       
            nion_sum_snapshot = comm.reduce(sum_nion[model_number][snapshot_idx], op = MPI.SUM, root = 0)
            if rank == 0:
                sum_array[model_number].append(nion_sum_snapshot * 1.0e50 / (pow(AllVars.BoxSize / AllVars.Hubble_h,3) * (float(LastFile[model_number] - FirstFile[model_number] + 1) / float(NumFiles[model_number]))))
            
    if (rank == 0):
        ax1 = plt.subplot(111)

        for model_number in range(0, len(SnapList)):

            if(simulation_norm[model_number] == 0):
                cosmo = AllVars.Set_Params_Mysim()
            if(simulation_norm[model_number] == 1):
                cosmo = AllVars.Set_Params_MiniMill()
            elif(simulation_norm[model_number] == 3):
                cosmo = AllVars.Set_Params_Tiamat_extended()
            elif(simulation_norm[model_number] == 4):
                cosmo = AllVars.Set_Params_Britton()
            elif(simulation_norm[model_number] == 5):
                cosmo = AllVars.Set_Params_Kali()
            else: 
                print("Simulation norm was set to {0}.".format(simulation_norm[model_number]))
                raise ValueError("This option has been implemented yet.  Get your head in the game Jacob!")


            t = np.empty(len(SnapList[model_number]))
            for snapshot_idx in range(0, len(SnapList[model_number])):
                t[snapshot_idx] = (AllVars.t_BigBang - cosmo.lookback_time(AllVars.SnapZ[SnapList[model_number][snapshot_idx]]).value) * 1.0e3     
    
           
            t = [t for t, N in zip(t, sum_array[model_number]) if N > 1.0]
            sum_array[model_number] = [x for x in sum_array[model_number] if x > 1.0]
                     
            print("The total number of ionizing photons for model {0} is {1} s^1 Mpc^-3".format(model_number, sum(sum_array[model_number])))    
            print(np.log10(sum_array[model_number])) 
            ax1.plot(t, np.log10(sum_array[model_number]), color = PlotScripts.colors[model_number], linestyle = PlotScripts.linestyles[model_number], label = model_tags[model_number], linewidth = PlotScripts.global_linewidth)  
            #ax1.fill_between(t, np.subtract(mean,std), np.add(mean,std), color = colors[model_number], alpha = 0.25)

        ax1.xaxis.set_minor_locator(mtick.MultipleLocator(PlotScripts.time_tickinterval))
        #ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.025))
        ax1.set_xlim(PlotScripts.time_xlim)
        ax1.set_ylim([48.5, 51.5])

        ax2 = ax1.twiny()

        t_plot = (AllVars.t_BigBang - cosmo.lookback_time(PlotScripts.z_plot).value) * 1.0e3 # Corresponding Time values on the bottom.
        z_labels = ["$%d$" % x for x in PlotScripts.z_plot] # Properly Latex-ize the labels.

        ax2.set_xlabel(r"$z$", size = PlotScripts.global_labelsize)
        ax2.set_xlim(PlotScripts.time_xlim)
        ax2.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
        ax2.set_xticklabels(z_labels) # But label them as redshifts.

        ax1.set_xlabel(r"$\mathrm{Time \: Since \: Big \: Bang \: [Myr]}$", size = PlotScripts.global_fontsize)
        ax1.set_ylabel(r'$\sum f_\mathrm{esc}\dot{N}_\gamma \: [\mathrm{s}^{-1}\mathrm{Mpc}^{-3}]$', fontsize = PlotScripts.global_fontsize) 

        plot_time = 1
        bouwens_z = np.arange(6,16) # Redshift range for the observations.
        bouwens_t = (AllVars.t_BigBang - cosmo.lookback_time(bouwens_z).value) * 1.0e3 # Corresponding values for what we will plot on the x-axis.

        bouwens_1sigma_lower = [50.81, 50.73, 50.60, 50.41, 50.21, 50.00, 49.80, 49.60, 49.39, 49.18] # 68% Confidence Intervals for the ionizing emissitivity from Bouwens 2015.
        bouwens_1sigma_upper = [51.04, 50.85, 50.71, 50.62, 50.56, 50.49, 50.43, 50.36, 50.29, 50.23]

        bouwens_2sigma_lower = [50.72, 50.69, 50.52, 50.27, 50.01, 49.75, 49.51, 49.24, 48.99, 48.74] # 95% CI. 
        bouwens_2sigma_upper = [51.11, 50.90, 50.74, 50.69, 50.66, 50.64, 50.61, 50.59, 50.57, 50.55]

        if plot_time == 1:
            ax1.fill_between(bouwens_t, bouwens_1sigma_lower, bouwens_1sigma_upper, color = 'k', alpha = 0.2)
            ax1.fill_between(bouwens_t, bouwens_2sigma_lower, bouwens_2sigma_upper, color = 'k', alpha = 0.4, label = r"$\mathrm{Bouwens \: et \: al. \: (2015)}$")
        else:
            ax1.fill_between(bouwens_z, bouwens_1sigma_lower, bouwens_1sigma_upper, color = 'k', alpha = 0.2)
            ax1.fill_between(bouwens_z, bouwens_2sigma_lower, bouwens_2sigma_upper, color = 'k', alpha = 0.4, label = r"$\mathrm{Bouwens \: et \: al. \: (2015)}$")

    #   ax1.text(0.075, 0.965, '(a)', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        ax1.text(350, 50.0, r"$68\%$", horizontalalignment='center', verticalalignment = 'center', fontsize = PlotScripts.global_labelsize) 
        ax1.text(350, 50.8, r"$95\%$", horizontalalignment='center', verticalalignment = 'center', fontsize = PlotScripts.global_labelsize)

        leg = ax1.legend(loc='lower right', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(PlotScripts.global_legendsize)

        plt.tight_layout()
        outputFile = './{0}{1}'.format(output_tag, output_format)
        plt.savefig(outputFile)  # Save the figure
        print('Saved file to {0}'.format(outputFile))
        plt.close()

##

def plot_singleSFR(galaxies_filepath_array, merged_galaxies_filepath_array, number_snapshots, simulation_norm, model_tags, output_tag):
 
    SFR_gal = []
    SFR_ensemble = []

    ejected_gal = []
    ejected_ensemble = []

    infall_gal = []
    infall_ensemble = [] 

    ejectedmass_gal = []
    ejectedmass_ensemble = []

    N_random = 1

    ax1 = plt.subplot(111)
#    ax3 = plt.subplot(122)
    #ax5 = plt.subplot(133)

    look_for_alive = 1
    #idx_array = [20004, 20005, 20016]
    #halonr_array = [7381]
    halonr_array = [389106]
    #halonr_array = [36885]
    for model_number in range(0, len(model_tags)):
        if(simulation_norm[model_number] == 0):
            AllVars.Set_Params_Mysim()
        if(simulation_norm[model_number] == 1):
            AllVars.Set_Params_MiniMill()
        elif(simulation_norm[model_number] == 3):
            AllVars.Set_Params_Tiamat_extended()
        else:
            print("Simulation norm was set to {0}.".format(simulation_norm[model_number]))
            raise ValueError("This option has been implemented yet.  Get your head in the game Jacob!")

        SFR_gal.append([])
        SFR_ensemble.append([])

        ejected_gal.append([])
        ejected_ensemble.append([])

        infall_gal.append([])
        infall_ensemble.append([])

        ejectedmass_gal.append([])
        ejectedmass_ensemble.append([])

        GG, Gal_Desc = ReadScripts.ReadGals_SAGE_DelayedSN(galaxies_filepath_array[model_number], 0, number_snapshots[model_number], comm) # Read in the correct galaxy file. 
        G_Merged, Merged_Desc = ReadScripts.ReadGals_SAGE_DelayedSN(merged_galaxies_filepath_array[model_number], 0, number_snapshots[model_number], comm) # Also need the merged galaxies.
        G = ReadScripts.Join_Arrays(GG, G_Merged, Gal_Desc) # Then join them together for all galaxies that existed at this Redshift. 


        if look_for_alive == 1:
            G.GridHistory[G.GridHistory >= 0] = 1
            G.GridHistory[G.GridHistory < 0] = 0
            alive = np.sum(G.GridHistory, axis = 1)
#            print "The galaxy that was present in the most snapshots is %d which was in %d snaps" %(np.argmax(alive), np.amax(alive)) 
            most_alive = alive.argsort()[-10:][::-1] # Finds the 3 galaxies alive for the most snapshots.  Taken from https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
#            print G.HaloNr[most_alive]
         

        t = np.empty((number_snapshots[model_number])) 
 
       
        for snapshot_idx in range(0, number_snapshots[model_number]): 
            w = np.where((G.GridHistory[:, snapshot_idx] != -1) & (G.GridStellarMass[:, snapshot_idx] > 0.0) & (G.GridStellarMass[:, snapshot_idx] < 1e5) & (G.GridFoFMass[:, snapshot_idx] >= m_low_SAGE) & (G.GridFoFMass[:, snapshot_idx] <=  m_high_SAGE))[0] # Only include those galaxies that existed at the current snapshot, had positive (but not infinite) stellar/Halo mass and Star formation rate.
            
            SFR_ensemble[model_number].append(np.mean(G.GridSFR[w,snapshot_idx]))
            ejected_ensemble[model_number].append(np.mean(G.GridOutflowRate[w, snapshot_idx]))
            infall_ensemble[model_number].append(np.mean(G.GridInfallRate[w, snapshot_idx]))

            t[snapshot_idx] = (t_BigBang - cosmo.lookback_time(AllVars.SnapZ[snapshot_idx]).value) * 1.0e3 
            
        for p in range(0, N_random):
            random_idx = (np.where((G.HaloNr == halonr_array[p]))[0])[0] 
            SFR_gal[model_number].append(G.GridSFR[random_idx]) # Remember the star formation rate history of the galaxy.
            ejected_gal[model_number].append(G.GridOutflowRate[random_idx])
            infall_gal[model_number].append(G.GridInfallRate[random_idx])
            ejectedmass_gal[model_number].append(G.GridEjectedMass[random_idx])
        
            #SFR_gal[model_number][p][SFR_gal[model_number][p] < 1.0e-15] = 1 
            for snapshot_idx in range(0, number_snapshots[model_number]):  
                if snapshot_idx == 0:
                    pass 
                elif(G.GridHistory[random_idx, snapshot_idx] == -1):
                    SFR_gal[model_number][p][snapshot_idx] = SFR_gal[model_number][p][snapshot_idx - 1]

#        SFR_ensemble[model_number] = np.nan_to_num(SFR_ensemble[model_number])        
#        SFR_ensemble[model_number][SFR_ensemble[model_number] < 1.0e-15] = 1    

         
#        ejected_ensemble[model_number][ejected_ensemble[model_number] < 1.0e-15] = 1     
       
        
        ax1.plot(t, SFR_ensemble[model_number], color = PlotScripts.colors[0], linestyle = PlotScripts.linestyles[model_number], label = model_tags[model_number], linewidth = PlotScripts.global_linewidth)
        ax1.plot(t, ejected_ensemble[model_number], color = PlotScripts.colors[1], linestyle = PlotScripts.linestyles[model_number], linewidth = PlotScripts.global_linewidth, alpha = 1.0)
        #ax5.plot(t, infall_ensemble[model_number], color = PlotScripts.colors[2], linestyle = PlotScripts.linestyles[model_number], linewidth = PlotScripts.global_linewidth, alpha = 1.0)
        #ax5.plot(t, ejectedmass_ensemble[model_number], color = PlotScripts.colors[2], linestyle = PlotScripts.linestyles[model_number], linewidth = PlotScripts.global_linewidth, alpha = 1.0)
        
        for p in range(0, N_random):
            ax1.plot(t, SFR_gal[model_number][p], color = PlotScripts.colors[0], linestyle = PlotScripts.linestyles[model_number], alpha = 0.5, linewidth = 1)
            ax1.plot(t, ejected_gal[model_number][p], color = PlotScripts.colors[1], linestyle = PlotScripts.linestyles[model_number], alpha = 0.5, linewidth = 1)
            #ax5.plot(t, infall_gal[model_number][p], color = PlotScripts.colors[2], linestyle = PlotScripts.linestyles[model_number], alpha = 0.5, linewidth = 1)
            #ax5.plot(t, ejectedmass_gal[model_number][p], color = PlotScripts.colors[2], linestyle = PlotScripts.linestyles[model_number], alpha = 0.5, linewidth = 1)

            #ax1.plot(t, SFR_gal[model_number][p], color = PlotScripts.colors[0], linestyle = PlotScripts.linestyles[model_number], alpha = 1.0, linewidth = 1, label = model_tags[model_number])
            #ax1.plot(t, ejected_gal[model_number][p], color = PlotScripts.colors[1], linestyle = PlotScripts.linestyles[model_number], alpha = 1.0, linewidth = 1, label = model_tags[model_number])

    ax1.plot(np.nan, np.nan, color = 'r', linestyle = '-', label = "SFR")
    ax1.plot(np.nan, np.nan, color = 'b', linestyle = '-', label = "Outflow")

#    exit() 
    #ax1.plot(np.nan, np.nan, color = PlotScripts.colors[0], label = 'SFR')
    #ax1.plot(np.nan, np.nan, color = PlotScripts.colors[1], label = 'Outflow')

    ax1.set_yscale('log', nonposy='clip')
    ax1.set_ylabel(r"$\mathrm{Mass \: Flow} \: [\mathrm{M}_\odot \mathrm{yr}^{-1}]$")
    ax1.set_xlabel(r"$\mathrm{Time \: Since \: Big \: Bang \: [Myr]}$", size = PlotScripts.global_fontsize)
    ax1.set_xlim(PlotScripts.time_xlim)
    ax1.set_ylim([1e-6, 1e3])

    '''
    ax3.set_yscale('log', nonposy='clip')
    ax3.set_ylabel(r"$\mathrm{Outflow \: Rate} \: [\mathrm{M}_\odot \mathrm{yr}^{-1}]$")
    ax3.set_xlabel(r"$\mathrm{Time \: Since \: Big \: Bang \: [Myr]}$", size = PlotScripts.global_fontsize)
    ax3.set_xlim(PlotScripts.time_xlim)
    ax3.set_ylim([1e-8, 1e3])

    ax5.set_yscale('log', nonposy='clip')
    #ax5.set_ylabel(r"$\mathrm{Infall \: Rate} \: [\mathrm{M}_\odot \mathrm{yr}^{-1}]$")
    ax5.set_ylabel(r"$\mathrm{Ejected Mass} [\mathrm{M}_\odot]$")
    ax5.set_xlabel(r"$\mathrm{Time \: Since \: Big \: Bang \: [Myr]}$", size = PlotScripts.global_fontsize)
    ax5.set_xlim(PlotScripts.time_xlim)
    #ax5.set_ylim([1e-8, 1e3])
    ax5.set_ylim([1e6, 1e10])
    '''
    ax2 = ax1.twiny()
    #ax4 = ax3.twiny()
    #ax6 = ax5.twiny()

    t_plot = (t_BigBang - cosmo.lookback_time(PlotScripts.z_plot).value) * 1.0e3 # Corresponding Time values on the bottom.
    z_labels = ["$%d$" % x for x in PlotScripts.z_plot] # Properly Latex-ize the labels.

    ax2.set_xlabel(r"$z$", size = PlotScripts.global_labelsize)
    ax2.set_xlim(PlotScripts.time_xlim)
    ax2.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
    ax2.set_xticklabels(z_labels) # But label them as redshifts.

    '''
    ax4.set_xlabel(r"$z$", size = PlotScripts.global_labelsize)
    ax4.set_xlim(PlotScripts.time_xlim)
    ax4.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
    ax4.set_xticklabels(z_labels) # But label them as redshifts.

    ax6.set_xlabel(r"$z$", size = PlotScripts.global_labelsize)
    ax6.set_xlim(PlotScripts.time_xlim)
    ax6.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
    ax6.set_xticklabels(z_labels) # But label them as redshifts.
    '''

    plt.tight_layout()
    leg = ax1.legend(loc='lower right', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize(PlotScripts.global_legendsize)


    outputFile = './Halo%d_mlow%.2f_%s%s' %(halonr_array[0], m_low_SAGE, output_tag, output_format) 
    plt.savefig(outputFile, bbox_inches='tight')  # Save the figure
    print('Saved file to {0}'.format(outputFile))
    plt.close()

##

def plot_quasars_count(SnapList, PlotList, N_quasars_z, N_quasars_boost_z, N_gal_z, mean_quasar_activity, std_quasar_activity, N_halo, N_merger_halo, N_gal, N_merger_galaxy, fesc_prescription, simulation_norm, FirstFile, LastFile, NumFile, model_tags, output_tag):
    '''
    Parameters
    ---------
    SnapList : Nested 'array-like` of ints, SnapList[model_number0] = [snapshot0_model0, ..., snapshotN_model0], with length equal to the number of models.
        Snapshots that we plot the quasar density at for each model.
    PlotList : Nested array of ints, PlotList[model_number0]= [plotsnapshot0_model0, ..., plotsnapshotN_model0], with length equal to the number of models.
        Snapshots that will be plotted for the quasar activity as a function of halo mass.
    N_quasars_z : Nested array of floats, N_quasars_z[model_number0] = [N_quasars_z0, N_quasars_z1, ..., N_quasars_zN]. Outer array has length equal to the number of models, inner array has length equal to length of the model's SnapList. 
        Number of quasars, THAT WENT OFF, during the given redshift. 
    N_quasars_boost_z : Nested array of floats, N_quasars_boost_z[model_number0] = [N_quasars_boost_z0, N_quasars_boost_z1, ..., N_quasars_boost_zN]. Outer array has length equal to the number of models, inner array has length equal to length of the model's SnapList. 
        Number of galaxies that had their escape fraction boosted by quasar activity. 
    N_gal_z : Nested array of floats, N_gal_z[model_number0] = [N_gal_z0, N_gal_z1, ..., N_gal_zN]. Outer array has length equal to the number of models, inner array has length equal to length of the model's SnapList. 
        Number of galaxies at each redshift.
    mean_quasar_activity, std_quasar_activity : Nested 2-dimensional array of floats, mean_quasar_activity[model_number0][snapshot0] = [bin0quasar_activity, ..., binNquasar_activity].  Outer array has length equal to the number of models, inner array has length equal to the length of the model's snaplist and most inner array has length equal to the number of halo bins (NB).
        Mean/std fraction of galaxies that had quasar go off during each snapshot as a function of halo mass.
        NOTE : This is for quasars going off, not for galaxies that have their escape fraction being boosted.
     
    fesc_prescription : Array with length equal to the number of models.
        Denotes what escape fraction prescription each model used.  Quasars are only tracked when fesc_prescription == 3.
    simulation_norm : array with length equal to the number of models.
        Denotes which simulation each model uses.  
        0 : MySim
        1 : Mini-Millennium
        2 : Tiamat (down to z = 5)
        3 : Extended Tiamat (down to z = 1.6ish).
        4 : Britton's Simulation
        5 : Kali
    FirstFile, LastFile, NumFile : array of integers with length equal to the number of models.
        The file numbers for each model that were read in (defined by the range between [FirstFile, LastFile] inclusive) and the TOTAL number of files for this model (we may only be plotting a subset of the volume). 
    model_tags : array of strings with length equal to the number of models.
        Strings that contain the tag for each model.  Will be placed on the plot.
    output_tag : string
        Name of the file that will be generated. File will be saved in the current directory with the output format defined by the 'output_format' variable at the beggining of the file.

    Returns
    -------
    No returns.
    Generates and saves the plot (named via output_tag).

    Units
    -----
    No relevant units. 
    '''

    print("Plotting quasar count/density")

    if rank == 0:       
        fig = plt.figure() 
        ax1 = fig.add_subplot(111)  
        ax6 = ax1.twinx()

        fig2 = plt.figure() 
        ax3 = fig2.add_subplot(111)  
        ax5 = ax3.twinx()       

        fig3 = plt.figure()
        ax7 = fig3.add_subplot(111)
   
        fig4 = plt.figure()
        ax50 = fig4.add_subplot(111)

        fig5 = plt.figure()
        ax55 = fig5.add_subplot(111)

        fig6 = plt.figure()
        ax56 = fig6.add_subplot(111)
 
    mean_quasar_activity_array = []
    std_quasar_activity_array = []
    N_quasar_activity_array = []

    N_gal_halo_array = []
    N_gal_array = []
 
    merger_counts_halo_array = []
    merger_counts_galaxy_array = []

    bin_middle_halo_array = []    
    bin_middle_galaxy_array = []    
 
    for model_number in range(0, len(SnapList)): # Does this for each of the models. 
        if (fesc_prescription[model_number] != 3): # Want to skip the models that didn't count quasars.
            continue
        ## Normalization for each model. ##
        if (simulation_norm[model_number] == 0):
            AllVars.Set_Params_Mysim()
        elif (simulation_norm[model_number] == 1):
            AllVars.Set_Params_MiniMill()
        elif (simulation_norm[model_number] == 2):
            AllVars.Set_Params_Tiamat()
        elif (simulation_norm[model_number] == 3):
            AllVars.Set_Params_Tiamat_extended()
        elif (simulation_norm[model_number] == 4):
            AllVars.Set_Params_Britton()       
        elif (simulation_norm[model_number] == 5):
            AllVars.Set_Params_Kali()

        mean_quasar_activity_array.append([])
        std_quasar_activity_array.append([])
        N_quasar_activity_array.append([])

        N_gal_halo_array.append([])
        N_gal_array.append([])

        merger_counts_halo_array.append([]) 
        merger_counts_galaxy_array.append([]) 

        bin_middle_halo_array.append([])
        bin_middle_galaxy_array.append([])
 
        box_factor = (LastFile[model_number] - FirstFile[model_number] + 1.0)/(NumFile[model_number]) # This factor allows us to take a sub-volume of the box and scale the results to represent the entire box.
        print("We are plotting the quasar density using {0:.4f} of the box's volume.".format(box_factor))
        norm = pow(AllVars.BoxSize,3) / pow(AllVars.Hubble_h, 3) * box_factor 

        ####    

        ## We perform the plotting on Rank 0 so only this rank requires the final counts array. ##
        if rank == 0:
            quasars_total = np.zeros_like((N_quasars_z[model_number]))
            boost_total = np.zeros_like(N_quasars_boost_z[model_number])
            gal_count_total = np.zeros_like(N_gal_z[model_number])
    
        else:
            quasars_total = None
            boost_total = None
            gal_count_total = None
       
        N_quasars_tmp = np.array((N_quasars_z[model_number])) # So we can use MPI.Reduce()
        comm.Reduce([N_quasars_tmp, MPI.DOUBLE], [quasars_total, MPI.DOUBLE], op = MPI.SUM, root = 0) # Sum the number of quasars and passes back to rank 0. 

        N_quasars_boost_tmp = np.array(N_quasars_boost_z[model_number]) # So we can use MPI.Reduce()
        comm.Reduce([N_quasars_boost_tmp, MPI.DOUBLE], [boost_total, MPI.DOUBLE], op = MPI.SUM, root = 0) # Sum the number of galaxies that had their fesc boosted. 

        N_gal_tmp = np.array(N_gal_z[model_number]) # So we can use MPI.Reduce()
        comm.Reduce([N_gal_tmp, MPI.DOUBLE], [gal_count_total, MPI.DOUBLE], op = MPI.SUM, root = 0) # Sum the number of total galaxies.

        for snapshot_idx in range(len(SnapList[model_number])): 
            mean_quasar_activity_array[model_number], std_quasar_activity_array[model_number], N_quasar_activity_array[model_number] = calculate_pooled_stats(mean_quasar_activity_array[model_number], std_quasar_activity_array[model_number], N_quasar_activity_array[model_number], mean_quasar_activity[model_number][snapshot_idx], std_quasar_activity[model_number][snapshot_idx], N_halo[model_number][snapshot_idx]) 

            if rank == 0:
                merger_count_halo_total = np.zeros_like((N_merger_halo[model_number][snapshot_idx]))
                N_gal_halo_total = np.zeros_like((N_halo[model_number][snapshot_idx]))

                merger_count_galaxy_total = np.zeros_like((N_merger_galaxy[model_number][snapshot_idx]))
                N_gal_total = np.zeros_like((N_gal[model_number][snapshot_idx]))
            else:
                merger_count_halo_total = None
                N_gal_halo_total = None
 
                merger_count_galaxy_total = None 
                N_gal_total = None
 
            comm.Reduce([N_merger_halo[model_number][snapshot_idx], MPI.FLOAT], [merger_count_halo_total, MPI.FLOAT], op = MPI.SUM, root = 0) # Sum all the stellar mass and pass to Rank 0.
            comm.Reduce([N_halo[model_number][snapshot_idx], MPI.FLOAT], [N_gal_halo_total, MPI.FLOAT], op = MPI.SUM, root = 0) # Sum all the stellar mass and pass to Rank 0.

            comm.Reduce([N_merger_galaxy[model_number][snapshot_idx], MPI.FLOAT], [merger_count_galaxy_total, MPI.FLOAT], op = MPI.SUM, root = 0) # Sum all the stellar mass and pass to Rank 0.
            comm.Reduce([N_gal[model_number][snapshot_idx], MPI.FLOAT], [N_gal_total, MPI.FLOAT], op = MPI.SUM, root = 0) # Sum all the stellar mass and pass to Rank 0.

            if rank == 0:
                merger_counts_halo_array[model_number].append(merger_count_halo_total)
                N_gal_halo_array[model_number].append(N_gal_halo_total)   

                merger_counts_galaxy_array[model_number].append(merger_count_galaxy_total)
                N_gal_array[model_number].append(N_gal_total)   
 
            bin_middle_halo_array[model_number].append(np.arange(m_low, m_high+bin_width, bin_width)[:-1] + bin_width * 0.5)
            bin_middle_galaxy_array[model_number].append(np.arange(m_gal_low, m_gal_high+bin_width, bin_width)[:-1] + bin_width * 0.5)
             
        if rank == 0:     
            plot_count = 0
            stop_plot = 0
            title = model_tags[model_number] 
            t = np.empty(len(SnapList[model_number]))
            ZZ = np.empty(len(SnapList[model_number]))
            for snapshot_idx in range(0, len(SnapList[model_number])):  
                t[snapshot_idx] = (AllVars.t_BigBang - AllVars.Lookback_Time[SnapList[model_number][snapshot_idx]]) * 1.0e3 
                ZZ[snapshot_idx] = AllVars.SnapZ[SnapList[model_number][snapshot_idx]] 
                if (stop_plot == 0):
#                    print("Snapshot {0} PlotSnapshot " 
#"{1}".format(SnapList[model_number][snapshot_idx], PlotList[model_number][plot_count]))
                    if (SnapList[model_number][snapshot_idx] == PlotList[model_number][plot_count]): 
                        label = "z = {0:.2f}".format(AllVars.SnapZ[PlotList[model_number][plot_count]])

                        ax7.plot(bin_middle_halo_array[model_number][snapshot_idx], mean_quasar_activity_array[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = label, linewidth = PlotScripts.global_linewidth)

                        #ax50.plot(bin_middle_halo_array[model_number][snapshot_idx], merger_counts_array[model_number][snapshot_idx] / gal_count_total[snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = label, linewidth = PlotScripts.global_linewidth)
                        ax50.plot(bin_middle_halo_array[model_number][snapshot_idx], merger_counts_halo_array[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = label, linewidth = PlotScripts.global_linewidth)
                        #ax50.plot(bin_middle_halo_array[model_number][snapshot_idx], merger_counts_array[model_number][snapshot_idx] / N_gal_halo_array[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = label, linewidth = PlotScripts.global_linewidth)
                                        
                        #ax55.plot(bin_middle_galaxy_array[model_number][snapshot_idx], merger_counts_galaxy_array[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = label, linewidth = PlotScripts.global_linewidth)
                        ax55.plot(bin_middle_galaxy_array[model_number][snapshot_idx],
merger_counts_galaxy_array[model_number][snapshot_idx] / N_gal_array[model_number][snapshot_idx], color = PlotScripts.colors[plot_count], linestyle = PlotScripts.linestyles[model_number], rasterized = True, label = label, linewidth = PlotScripts.global_linewidth)
                        print("plot_count = {0} len(PlotList) = {1}".format(plot_count,
len(PlotList[model_number])))
                        plot_count += 1
                        print("plot_count = {0} len(PlotList) = {1}".format(plot_count,
len(PlotList[model_number])))
                        if (plot_count == len(PlotList[model_number])):
                            stop_plot = 1
                print("For Snapshot {0} at t {3} there were {1} total mergers compared to {2} total galaxies.".format(snapshot_idx, np.sum(merger_counts_galaxy_array[model_number][snapshot_idx]), np.sum(gal_count_total[snapshot_idx]), t[snapshot_idx]))
                if (np.sum(gal_count_total[snapshot_idx]) > 0.0 and np.sum(merger_counts_galaxy_array[model_number][snapshot_idx]) > 0.0):
                    
                    ax56.scatter(t[snapshot_idx], np.sum(merger_counts_galaxy_array[model_number][snapshot_idx]) / np.sum(gal_count_total[snapshot_idx]), color = 'r', rasterized = True)
                    #ax56.scatter(t[snapshot_idx], quasars_total[snapshot_idx] / np.sum(gal_count_total[snapshot_idx]), color = 'r', rasterized = True)
            ax1.plot(t, quasars_total / norm, color = PlotScripts.colors[model_number], linestyle = PlotScripts.linestyles[0], rasterized = True, linewidth = PlotScripts.global_linewidth)
           
            p = np.where((ZZ < 15))[0] 
            #ax1.plot(ZZ[p], quasars_total[p] / norm, color = PlotScripts.colors[model_number], linestyle = PlotScripts.linestyles[0], rasterized = True, linewidth = PlotScripts.global_linewidth)
             
            ax3.plot(t, boost_total, color = PlotScripts.colors[model_number], linestyle = PlotScripts.linestyles[0], rasterized = True, label = title, linewidth = PlotScripts.global_linewidth) 
            w = np.where((gal_count_total > 0.0))[0] # Since we're doing a division, need to only plot those redshifts that actually have galaxies.

            ax5.plot(t[w], np.divide(boost_total[w], gal_count_total[w]), color = PlotScripts.colors[model_number], linestyle = PlotScripts.linestyles[1], rasterized = True, linewidth = PlotScripts.global_linewidth) 
            ax6.plot(t[w], gal_count_total[w] / norm, color = PlotScripts.colors[model_number], linestyle = PlotScripts.linestyles[1], rasterized = True, linewidth = PlotScripts.global_linewidth)
            #ax6.plot(ZZ[p], gal_count_total[p] / norm, color = PlotScripts.colors[model_number], linestyle = PlotScripts.linestyles[1], rasterized = True, linewidth = PlotScripts.global_linewidth)
            
            ax1.plot(np.nan, np.nan, color = PlotScripts.colors[0], linestyle = PlotScripts.linestyles[0], label = "Quasar Ejection Density")
            ax1.plot(np.nan, np.nan, color = PlotScripts.colors[0], linestyle = PlotScripts.linestyles[1], label = "Galaxy Density")

            ax3.plot(np.nan, np.nan, color = 'k', linestyle = PlotScripts.linestyles[0], label = "Count")
            ax3.plot(np.nan, np.nan, color = 'k', linestyle = PlotScripts.linestyles[1], label = "Fraction of Galaxies")
           
            ax7.set_xlabel(r'$\log_{10}\ M_\mathrm{vir}\ [M_{\odot}]$', size = PlotScripts.global_fontsize) 
            ax7.set_ylabel(r'$\mathrm{Mean \: Quasar \: Activity}$', size = PlotScripts.global_fontsize)
       
            ax50.set_xlabel(r'$\log_{10}\ M_\mathrm{vir}\ [M_{\odot}]$', size = PlotScripts.global_fontsize) 
            #ax50.set_ylabel(r'$\mathrm{Fraction \: Galaxies \: Undergoing \: Merger}$', size = PlotScripts.global_fontsize)
            ax50.set_ylabel(r'$\mathrm{Number \: Galaxies \: Undergoing \: Merger}$', size = PlotScripts.global_fontsize)

            ax55.set_xlabel(r'$\log_{10}\ M_\mathrm{*}\ [M_{\odot}]$', size = PlotScripts.global_fontsize) 
            ax55.set_ylabel(r'$\mathrm{Fraction \: Galaxies \: Undergoing \: Merger}$', size = PlotScripts.global_fontsize)
            #ax55.set_ylabel(r'$\mathrm{Number \: Galaxies \: Undergoing \: Merger}$', size = PlotScripts.global_fontsize)

            ax56.set_xlabel(r"$\mathrm{Time \: Since \: Big \: Bang \: [Myr]}$", size = PlotScripts.global_labelsize)             
            ax56.set_ylabel(r'$\mathrm{Fraction \: Galaxies \: Undergoing \: Merger}$', size = PlotScripts.global_fontsize)
            #ax56.set_ylabel(r'$\mathrm{Fraction \: Galaxies \: Quasar \: Activity}$', size = PlotScripts.global_fontsize)
            ax56.set_yscale('log', nonposy='clip')

            ax50.axvline(np.log10(32.0*AllVars.PartMass / AllVars.Hubble_h), color = 'k', linewidth = PlotScripts.global_linewidth, linestyle = '-.')   
         
            ax1.xaxis.set_minor_locator(mtick.MultipleLocator(PlotScripts.time_tickinterval))
            ax1.set_xlim(PlotScripts.time_xlim)
            ax1.set_yscale('log', nonposy='clip')

            ax3.xaxis.set_minor_locator(mtick.MultipleLocator(PlotScripts.time_tickinterval))
            ax3.set_xlim(PlotScripts.time_xlim)
            ax3.set_yscale('log', nonposy='clip')

            ## Create a second axis at the top that contains the corresponding redshifts. ##
            ## The redshift defined in the variable 'z_plot' will be displayed. ##
            ax2 = ax1.twiny()
            ax4 = ax3.twiny()
            ax57 = ax56.twiny()

            t_plot = (AllVars.t_BigBang - AllVars.cosmo.lookback_time(PlotScripts.z_plot).value) * 1.0e3 # Corresponding time values on the bottom.
            z_labels = ["$%d$" % x for x in PlotScripts.z_plot] # Properly Latex-ize the labels.

            ax2.set_xlabel(r"$z$", size = PlotScripts.global_labelsize) 
            ax2.set_xlim(PlotScripts.time_xlim)
            ax2.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
            ax2.set_xticklabels(z_labels) # But label them as redshifts.

            ax4.set_xlabel(r"$z$", size = PlotScripts.global_labelsize) 
            ax4.set_xlim(PlotScripts.time_xlim)
            ax4.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
            ax4.set_xticklabels(z_labels) # But label them as redshifts.

            ax57.set_xlabel(r"$z$", size = PlotScripts.global_labelsize) 
            ax57.set_xlim(PlotScripts.time_xlim)
            ax57.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
            ax57.set_xticklabels(z_labels) # But label them as redshifts.


            ax1.set_xlabel(r"$\mathrm{Time \: Since \: Big \: Bang \: [Myr]}$", size = PlotScripts.global_labelsize) 
            #ax1.set_xlabel(r"$z$", size = PlotScripts.global_labelsize) 
            ax1.set_ylabel(r'$N_\mathrm{Quasars} \: [\mathrm{Mpc}^{-3}]$', fontsize = PlotScripts.global_fontsize) 
            ax6.set_ylabel(r'$N_\mathrm{Gal} \: [\mathrm{Mpc}^{-3}]$', fontsize = PlotScripts.global_fontsize) 

            ax3.set_xlabel(r"$\mathrm{Time \: Since \: Big \: Bang \: [Myr]}$", size = PlotScripts.global_labelsize) 
            ax3.set_ylabel(r'$N_\mathrm{Boosted}$', fontsize = PlotScripts.global_fontsize) 
            ax5.set_ylabel(r'$\mathrm{Fraction \: Boosted}$', fontsize = PlotScripts.global_fontsize) 

            leg = ax1.legend(loc='lower right', numpoints=1, labelspacing=0.1)
            leg.draw_frame(False)  # Don't want a box frame
            for t in leg.get_texts():  # Reduce the size of the text
                t.set_fontsize(PlotScripts.global_legendsize)

            leg = ax3.legend(loc='lower left', numpoints=1, labelspacing=0.1)
            leg.draw_frame(False)  # Don't want a box frame
            for t in leg.get_texts():  # Reduce the size of the text
                t.set_fontsize(PlotScripts.global_legendsize)

            leg = ax7.legend(loc='upper left', numpoints=1, labelspacing=0.1)
            leg.draw_frame(False)  # Don't want a box frame
            for t in leg.get_texts():  # Reduce the size of the text
                t.set_fontsize(PlotScripts.global_legendsize)

            leg = ax50.legend(loc='upper right', numpoints=1, labelspacing=0.1)
            leg.draw_frame(False)  # Don't want a box frame
            for t in leg.get_texts():  # Reduce the size of the text
                t.set_fontsize(PlotScripts.global_legendsize)

            leg = ax55.legend(loc='upper right', numpoints=1, labelspacing=0.1)
            leg.draw_frame(False)  # Don't want a box frame
            for t in leg.get_texts():  # Reduce the size of the text
                t.set_fontsize(PlotScripts.global_legendsize)

            fig.tight_layout()            
            fig2.tight_layout()           
            fig3.tight_layout()           
            fig5.tight_layout()
            fig6.tight_layout()
 
            outputFile1 = './{0}_quasardensity{1}'.format(output_tag, output_format)
            outputFile2 = './{0}_boostedcount{1}'.format(output_tag, output_format)
            outputFile3 = './{0}_quasar_activity_halo{1}'.format(output_tag, output_format)
            outputFile4 = './{0}_mergercount_global{1}'.format(output_tag, output_format)
            outputFile5 = './{0}_mergercount_global_stellarmass{1}'.format(output_tag, output_format)
            outputFile6 = './{0}_mergercount_total{1}'.format(output_tag, output_format)

            fig.savefig(outputFile1)  # Save the figure
            fig2.savefig(outputFile2)  # Save the figure
            fig3.savefig(outputFile3)  # Save the figure
            fig4.savefig(outputFile4)  # Save the figure
            fig5.savefig(outputFile5)  # Save the figure
            fig6.savefig(outputFile6)  # Save the figure

            print("Saved to {0}".format(outputFile1))
            print("Saved to {0}".format(outputFile2))
            print("Saved to {0}".format(outputFile3))
            print("Saved to {0}".format(outputFile4))
            print("Saved to {0}".format(outputFile5))
            print("Saved to {0}".format(outputFile6))

            plt.close(fig)
            plt.close(fig2)
            plt.close(fig3)

##

def plot_photon_quasar_fraction(snapshot, filenr, output_tag, QuasarFractionalPhoton, QuasarActivityToggle, NumSubsteps):

    ax1 = plt.subplot(111)

    counts, bin_edges, bin_middle = AllVars.Calculate_Histogram(QuasarFractionalPhoton, 0.05, 0, 0, 1) 
            
    ax1.plot(bin_middle, counts, lw = PlotScripts.global_linewidth, color = 'r')
   
    ax1.axvline(np.mean(QuasarFractionalPhoton[QuasarFractionalPhoton != 0]), lw = 0.5, ls = '-')
 
    ax1.set_yscale('log', nonposy='clip')
    ax1.set_xlabel(r"$\mathrm{Fractional \: Photon \: Boost}$")
    ax1.set_ylabel(r"$\mathrm{Count}$")

    ax1.set_ylim([1e1, 1e5])

    outputFile1 = './photonfraction/file{0}_snap{1}_{2}{3}'.format(filenr, snapshot, output_tag, output_format)

    plt.tight_layout()
    plt.savefig(outputFile1)

    print("Saved to {0}".format(outputFile1))
    
    plt.close()
###

def plot_quasar_substep(snapshot, filenr, output_tag, substep):

    ax1 = plt.subplot(111)

    counts, bin_edges, bin_middle = AllVars.Calculate_Histogram(substep, 0.1, 0, 0, 10) 
            
    ax1.plot(bin_middle, counts, lw = PlotScripts.global_linewidth, color = 'r')
   
    ax1.axvline(np.mean(substep[substep != -1]), lw = 0.5, ls = '-')
 
    ax1.set_yscale('log', nonposy='clip')
    ax1.set_xlabel(r"$\mathrm{Substep \: Quasar \: Activity}$")
    ax1.set_ylabel(r"$\mathrm{Count}$")

#    ax1.set_ylim([1e1, 1e5])

    outputFile1 = './substep_activity/file{0}_snap{1}_{2}{3}'.format(filenr, snapshot, output_tag, output_format)

    plt.tight_layout()
    plt.savefig(outputFile1)

    print("Saved to {0}".format(outputFile1))
    
    plt.close()
 
###

def plot_post_quasar_SFR(PlotSnapList, model_number, Gal, output_tag):

    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()

    count = 0
    snapshot_thickness = 20 # How many snapshots before/after the quasar event do we want to track?
    for snapshot_idx in PlotSnapList[model_number]:
        w = np.where((G.QuasarActivity[:, snapshot_idx] == 1) & (G.LenHistory[:, snapshot_idx] > 200.0) & (G.GridStellarMass[:, snapshot_idx] > 0.001))[0]

        w_slice_gridhistory = G.GridHistory[w,snapshot_idx-snapshot_thickness:snapshot_idx+snapshot_thickness]
        
        potential_gal = [] 
        for i in range(len(w_slice_gridhistory)):
            ww = np.where((w_slice_gridhistory[i] >= 0))[0]
            if (len(ww) == snapshot_thickness * 2):
                potential_gal.append(w[i]) 
            
        if (len(potential_gal) == 0):
            return
        count += 1
        print("There were {0} galaxies that had an energetic quasar wind event at snapshot {1} (z = {2:.3f})".format(len(potential_gal), snapshot_idx, AllVars.SnapZ[snapshot_idx]))
        chosen_gal = potential_gal[1]

        lenhistory_array = np.empty((int(snapshot_thickness*2 + 1)))
        SFR_array = np.empty((int(snapshot_thickness*2 + 1)))
        gridhistory_array = np.empty((int(snapshot_thickness*2 + 1)))
        coldgas_array = np.empty((int(snapshot_thickness*2 + 1)))
        t = np.empty((int(snapshot_thickness*2 + 1)))
        for i in range(-snapshot_thickness, snapshot_thickness+1):
            #print("SFR {0} {1}".format(snapshot_idx + i, G.GridSFR[chosen_gal, snapshot_idx+i]))
            #print("ColdGas {0} {1}".format(snapshot_idx + i, G.GridColdGas[chosen_gal, snapshot_idx+i]))
            lenhistory_array[i+snapshot_thickness] = (G.LenHistory[chosen_gal, snapshot_idx+i]) 
            SFR_array[i+snapshot_thickness] = (G.GridSFR[chosen_gal, snapshot_idx+i]) #- (G.GridSFR[chosen_gal, snapshot_idx]) 
            gridhistory_array[i+snapshot_thickness] = (G.GridHistory[chosen_gal, snapshot_idx+i]) 
            coldgas_array[i+snapshot_thickness] = (G.GridColdGas[chosen_gal, snapshot_idx+i] * 1.0e10 / AllVars.Hubble_h) #- (G.GridColdGas[chosen_gal, snapshot_idx]) 
            t[i+snapshot_thickness] = (-AllVars.Lookback_Time[snapshot_idx+i] + AllVars.Lookback_Time[snapshot_idx]) * 1.0e3

        print("Len History {0}".format(lenhistory_array))
        print("Grid History {0}".format(gridhistory_array))
        print("Cold Gas {0}".format(coldgas_array))
        print("SFR {0}".format(SFR_array))

        stellarmass_text = r"$log M_* = {0:.2f} \: M_\odot$".format(np.log10(G.GridStellarMass[chosen_gal, snapshot_idx] * 1.0e10 / AllVars.Hubble_h))
        Ndym_text = "Dynamical Time = {0:.2f} Myr".format(G.DynamicalTime[chosen_gal, snapshot_idx])
        z_text = "z = {0:.2f}".format(AllVars.SnapZ[snapshot_idx])

        ax1.text(0.05, 0.95, z_text, transform = ax1.transAxes, fontsize = PlotScripts.global_fontsize - 4)
        ax1.text(0.05, 0.9, stellarmass_text, transform = ax1.transAxes, fontsize = PlotScripts.global_fontsize - 4)
        ax1.text(0.05, 0.85, Ndym_text, transform = ax1.transAxes, fontsize = PlotScripts.global_fontsize - 4)
        ax1.plot(t, SFR_array, color = 'r', lw = PlotScripts.global_linewidth) 
        ax2.plot(t, coldgas_array, color = 'b', lw = PlotScripts.global_linewidth)
        ax1.set_xlabel(r"$\mathrm{Time \: Since \: Quasar \: Event \: [Myr]}$", size = PlotScripts.global_labelsize - 10) 
#        ax1.set_ylabel(r"$\mathrm{Fractional \: SFR \: Relative \: To \: SFR_{Quasar}}$", size = PlotScripts.global_labelsize - 10) 
#        ax2.set_ylabel(r"$\mathrm{Difference \: Cold \: Gas \: Mass \: Relative \: To \: Cold_{Quasar}}$", size = PlotScripts.global_labelsize - 10) 
        ax1.set_ylabel(r"$\mathrm{SFR} \: [\mathrm{M}_\odot \mathrm{yr}^{-1}]$", size = PlotScripts.global_labelsize - 10) 
        ax2.set_ylabel(r"$\mathrm{Cold \: Gas \: Mass \: [\mathrm{M}_\odot]}$",size = PlotScripts.global_labelsize - 10) 

        ax1.set_yscale('log', nonposy='clip')
        ax2.set_yscale('log', nonposy='clip')
        ax1.plot(np.nan, np.nan, color = 'r', label = r"$\mathrm{SFR}$")
        ax1.plot(np.nan, np.nan, color = 'b', label = r"$\mathrm{Cold \: Gas}$")


        leg = ax1.legend(loc='upper right', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(PlotScripts.global_legendsize)

        
        outputFile = "{0}_galaxy{2}{1}".format(output_tag, output_format, chosen_gal)

        plt.tight_layout()
        plt.savefig(outputFile)

        print("Saved to {0}".format(outputFile))
        
        plt.close()
        exit()

###

def plot_stellarmass_blackhole(SnapList, simulation_norm, mean_galaxy_BHmass, std_galaxy_BHmass, N_galaxy_BHmass, model_tags, output_tag):

    mean_BHmass_stellar_array = []
    std_BHmass_stellar_array = []
    N_BHmass_stellar_array = []

    bin_middle_stellar_array = []
    redshift_labels = []

    for model_number in range(0, len(SnapList)):
        redshift_labels.append([])

        mean_BHmass_stellar_array.append([])
        std_BHmass_stellar_array.append([])
        N_BHmass_stellar_array.append([])

        bin_middle_stellar_array.append([])        

    for model_number in range(0, len(SnapList)): 

        ## Normalization for each model. ##
        if (simulation_norm[model_number] == 0):
            AllVars.Set_Params_Mysim()
        elif (simulation_norm[model_number] == 1):
            AllVars.Set_Params_MiniMill()
        elif (simulation_norm[model_number] == 2):
            AllVars.Set_Params_Tiamat()
        elif (simulation_norm[model_number] == 3):
            AllVars.Set_Params_Tiamat_extended()
        elif (simulation_norm[model_number] == 4):
            AllVars.Set_Params_Britton()       
        elif(simulation_norm[model_number] == 5):
            AllVars.Set_Params_Kali()

        for snapshot_idx in range(0, len(SnapList[model_number])):
            tmp = 'z = %.2f' %(AllVars.SnapZ[SnapList[model_number][snapshot_idx]])
            redshift_labels[model_number].append(tmp)
 
            mean_BHmass_stellar_array[model_number], std_BHmass_stellar_array[model_number], N_BHmass_stellar_array[model_number] = calculate_pooled_stats(mean_BHmass_stellar_array[model_number], std_BHmass_stellar_array[model_number], N_BHmass_stellar_array[model_number], mean_galaxy_BHmass[model_number][snapshot_idx], std_galaxy_BHmass[model_number][snapshot_idx], N_galaxy_BHmass[model_number][snapshot_idx]) 

            bin_middle_stellar_array[model_number].append(np.arange(m_gal_low, m_gal_high+bin_width, bin_width)[:-1] + bin_width * 0.5)
         
    if rank == 0:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        for model_number in range(0, len(SnapList)):
            for snapshot_idx in range(0, len(SnapList[model_number])):
                w = np.where((N_BHmass_stellar_array[model_number][snapshot_idx] > 0.0))[0]

                print(mean_BHmass_stellar_array[model_number][snapshot_idx][w])
                print(std_BHmass_stellar_array[model_number][snapshot_idx][w])
                print(N_BHmass_stellar_array[model_number][snapshot_idx][w])
                print(np.subtract(mean_BHmass_stellar_array[model_number][snapshot_idx][w], std_BHmass_stellar_array[model_number][snapshot_idx][w]))
                mean = np.log10(mean_BHmass_stellar_array[model_number][snapshot_idx][w])
                upper = np.log10(np.add(mean_BHmass_stellar_array[model_number][snapshot_idx][w], std_BHmass_stellar_array[model_number][snapshot_idx][w]))
                lower = np.log10(np.subtract(mean_BHmass_stellar_array[model_number][snapshot_idx][w], std_BHmass_stellar_array[model_number][snapshot_idx][w]))
                 
                ax1.plot(bin_middle_stellar_array[model_number][snapshot_idx][w], mean, label = redshift_labels[model_number][snapshot_idx], color = PlotScripts.colors[snapshot_idx], ls = PlotScripts.linestyles[model_number], lw = PlotScripts.global_linewidth, rasterized = True) 
                ax1.fill_between(bin_middle_stellar_array[model_number][snapshot_idx][w], lower, upper, color = PlotScripts.colors[model_number], alpha = 0.25)

        Obs.Get_Data_SMBH()
        PlotScripts.plot_SMBH_z8(ax1) 

        ax1.set_xlabel(r"$\log_{10}\mathrm{M}_* [\mathrm{M}_\odot]$", size = PlotScripts.global_fontsize) 
        ax1.set_ylabel(r"$\log_{10}\mathrm{M}_\mathrm{BH} [\mathrm{M}_\odot]$", size = PlotScripts.global_fontsize)     

        ax1.set_xticks(np.arange(7.0, 12.0))  
        ax1.set_yticks(np.arange(3.0, 12.0))  

        ax1.xaxis.set_minor_locator(mtick.MultipleLocator(0.25))
        ax1.yaxis.set_minor_locator(mtick.MultipleLocator(0.25))
        
        ax1.set_xlim([7.0, 10.25])
        ax1.set_ylim([3.0, 8.0])


        leg = ax1.legend(loc='upper left', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(PlotScripts.global_legendsize)
   
        outputFile = "{0}{1}".format(output_tag, output_format)

        plt.tight_layout()
        fig.savefig(outputFile)

        print("Saved to {0}".format(outputFile))
        
        plt.close()
        exit()


###

def plot_reionmod(PlotSnapList, SnapList, simulation_norm, mean_reionmod_halo, 
                  std_reionmod_halo, N_halo, mean_reionmod_z, std_reionmod_z, 
                  N_reionmod, plot_z, model_tags, output_tag): 
    """
    Plot the reionization modifier as a function of halo mass and redshift.
 
    Parameters
    ----------

    PlotSnapList, SnapList: 2D Nested arrays of integers. Outer length is equal to the number of models and inner length is number of snapshots we're plotting/calculated for. 
        PlotSnapList contains the snapshots for each model we will plot for the halo mass figure. 
        SnapList contains the snapshots for each model that we have performed calculations for.  These aren't equal because we don't want to plot halo curves for ALL redshifts.

    simulation_norm: Array of integers. Length is equal to the number of models.
        Contains the simulation identifier for each model. Used to set the parameters of each model.

    mean_reionmod_halo, std_reionmod_halo: 3D Nested arrays of floats.  Most outer length is equal to the number of models, next length is number of snapshots for each model, then inner-most length is the number of halo mass-                                                       bins (given by NB).
        Contains the mean/standard deviation values for the reionization modifier as a function of halo mass.   
        NOTE: These are unique for each task.
         
    N_halo: 3D Nested arrays of floats.  Lengths are identical to mean_reionmod_halo.
        Contains the number of halos in each halo mass bin.
        NOTE: These are unique for each task.

    mean_reionmod_z, std_reionmod_z: 2D Nested arrays of floats. Outer length is equal to the number of models, inner length is the number of snapshots for each model. NOTE: This inner length can be different to the length of                                                 PlotSnapList as we don't necessarily need to plot for every snapshot we calculate.
        Contains the mean/standard deviation values for the rieonization modifier as a function of redshift. 
        NOTE: These are unique for each task.

    N_reionmod: 2D Nested arrays of floats. Lengths are identical to mean_reionmod_z.
        Contains the number of galaxies at each redshift that have non-negative reionization modifier.  A negative reionization modifier is a galaxy who didn't have infall/stripping during the snapshot.
        NOTE: These are unique for each task.

    plot_z: Boolean.
        Denotes whether we want to plot the reionization modifier as a function
        of redshift. Useful because we often only calculate statistics for a
        subset of the snapshots to decrease computation time. For these runs,
        we don't want to plot for something that requires ALL snapshots. 

    model_tags: Array of strings. Length is equal to the number of models.
        Contains the legend labels for each model.

    output_tag: String.
        The prefix for the output file.

    Returns
    ----------
    
    None. Plot is saved in current directory as "./<output_tag>.<output_format>"

    """

    master_mean_reionmod_halo, master_std_reionmod_halo,
    master_N_reionmod_halo, master_bin_middle = collect_across_tasks(mean_reionmod_halo, 
                                                                     std_reionmod_halo, 
                                                                     N_halo, SnapList, 
                                                                     PlotSnapList, True, 
                                                                     m_low, m_high)
    if plot_z: 
        master_mean_reionmod_z, master_std_reionmod_z, master_N_reionmod_z, _ = collect_across_tasks(mean_reionmod_z, 
                                                                                                       std_reionmod_z, 
                                                                                                       N_reionmod)                                                                          

    if rank == 0:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        if plot_z: 
            fig2 = plt.figure()
            ax10 = fig2.add_subplot(111)

        for model_number in range(len(PlotSnapList)):        
            if(simulation_norm[model_number] == 1):
                cosmo = AllVars.Set_Params_MiniMill()
            elif(simulation_norm[model_number] == 3):
                cosmo = AllVars.Set_Params_Tiamat_extended()
            elif(simulation_norm[model_number] == 4):
                cosmo = AllVars.Set_Params_Britton()
            elif(simulation_norm[model_number] == 5):
                cosmo = AllVars.Set_Params_Kali()
    
            for snapshot_idx in range(len((PlotSnapList[model_number]))):
                if snapshot_idx == 0:
                    label = model_tags[model_number]
                else:
                    label = ""
                nonzero_bins = np.where(master_N_reionmod_halo[model_number][snapshot_idx] > 0.0)[0]
                ax1.plot(master_bin_middle[model_number][snapshot_idx][nonzero_bins], 
                         master_mean_reionmod_halo[model_number][snapshot_idx][nonzero_bins], 
                         label = label, ls = PlotScripts.linestyles[model_number], 
                         color = PlotScripts.colors[snapshot_idx]) 

            if plot_z:
                ax10.plot((AllVars.t_BigBang - AllVars.Lookback_Time[SnapList[model_number]])*1.0e3, master_mean_reionmod_z[model_number], color = PlotScripts.colors[model_number], label = model_tags[model_number], ls = PlotScripts.linestyles[model_number], lw = 3)

        for count, snapshot_idx in enumerate(PlotSnapList[model_number]):
            #label = r"$\mathbf{z = " + str(int(round(AllVars.SnapZ[snapshot_idx]))) + "}$"                
            label = r"$\mathbf{z = " + str(AllVars.SnapZ[snapshot_idx]) + "}$"                
            ax1.plot(np.nan, np.nan, ls = PlotScripts.linestyles[0], color =
                     PlotScripts.colors[count], label = label) 

        ax1.set_xlim([8.5, 11.5])
        ax1.set_ylim([0.0, 1.05])

        ax1.set_xlabel(r'$\mathbf{log_{10} \: M_{vir} \:[M_{\odot}]}$', fontsize = PlotScripts.global_labelsize) 
        ax1.set_ylabel(r'$\mathbf{Mean ReionMod}$', fontsize = PlotScripts.global_labelsize) 

        leg = ax1.legend(loc='lower right', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(PlotScripts.global_legendsize)

        outputFile1 = "./{0}_halo{1}".format(output_tag, output_format) 
        fig1.savefig(outputFile1, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile1))
        plt.close(fig1)


        if plot_z:
            ax10.set_xlabel(r"$\mathbf{Time \: since \: Big \: Bang \: [Myr]}$", fontsize = PlotScripts.global_labelsize)
            tick_locs = np.arange(200.0, 1000.0, 100.0)
            tick_labels = [r"$\mathbf{%d}$" % x for x in tick_locs]
            ax10.xaxis.set_major_locator(mtick.MultipleLocator(100))
            ax10.set_xticklabels(tick_labels, fontsize = PlotScripts.global_fontsize)
            ax10.set_xlim(PlotScripts.time_xlim)

            ax10.set_ylabel(r'$\mathbf{Mean ReionMod}$', fontsize = PlotScripts.global_labelsize)                 

            ax11 = ax10.twiny()

            t_plot = (AllVars.t_BigBang - cosmo.lookback_time(PlotScripts.z_plot).value) * 1.0e3 # Corresponding Time values on the bottom.
            z_labels = ["$\mathbf{%d}$" % x for x in PlotScripts.z_plot] # Properly Latex-ize the labels.

            ax11.set_xlabel(r"$\mathbf{z}$", fontsize = PlotScripts.global_labelsize)
            ax11.set_xlim(PlotScripts.time_xlim)
            ax11.set_xticks(t_plot) # Set the ticks according to the time values on the bottom,
            ax11.set_xticklabels(z_labels, fontsize = PlotScripts.global_fontsize) # But label them as redshifts.

            leg = ax10.legend(loc='lower right', numpoints=1, labelspacing=0.1)
            leg.draw_frame(False)  # Don't want a box frame
            for t in leg.get_texts():  # Reduce the size of the text
                t.set_fontsize(PlotScripts.global_legendsize)

            outputFile2 = "./{0}_z{1}".format(output_tag, output_format) 
            fig2.savefig(outputFile2, bbox_inches='tight')  # Save the figure
            print('Saved file to {0}'.format(outputFile2))
            plt.close(fig2)

##

def plot_dust(PlotSnapList, SnapList, simulation_norm, mean_dust_galaxy, std_dust_galaxy, 
              N_galaxy, mean_dust_halo, std_dust_halo, N_halo, plot_z, 
              model_tags, output_tag): 
    """

    """

    master_mean_dust_galaxy, master_std_dust_galaxy, master_N_dust_galaxy, master_bin_middle_galaxy = \
    collect_across_tasks(mean_dust_galaxy, std_dust_galaxy, N_galaxy, SnapList,
                         PlotSnapList, True, m_gal_low, m_gal_high)
    
    master_mean_dust_halo, master_std_dust_halo, master_N_dust_halo, master_bin_middle_halo = \
    collect_across_tasks(mean_dust_halo, std_dust_halo, N_halo, SnapList,
                         PlotSnapList, True, m_low, m_high)

    if rank == 0:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        for model_number in range(len(PlotSnapList)):        
            if(simulation_norm[model_number] == 1):
                cosmo = AllVars.Set_Params_MiniMill()
            elif(simulation_norm[model_number] == 3):
                cosmo = AllVars.Set_Params_Tiamat_extended()
            elif(simulation_norm[model_number] == 4):
                cosmo = AllVars.Set_Params_Britton()
            elif(simulation_norm[model_number] == 5):
                cosmo = AllVars.Set_Params_Kali()
    
            for snapshot_idx in range(len((PlotSnapList[model_number]))):
                if snapshot_idx == 0:
                    label = model_tags[model_number]
                else:
                    label = ""
                nonzero_bins = np.where(master_N_dust_galaxy[model_number][snapshot_idx] > 0.0)[0]
                ax1.plot(master_bin_middle_galaxy[model_number][snapshot_idx][nonzero_bins], 
                         master_mean_dust_galaxy[model_number][snapshot_idx][nonzero_bins], 
                         label = label, ls = PlotScripts.linestyles[model_number], 
                         color = PlotScripts.colors[snapshot_idx]) 

                nonzero_bins = np.where(master_N_dust_halo[model_number][snapshot_idx] > 0.0)[0]
                ax2.plot(master_bin_middle_halo[model_number][snapshot_idx][nonzero_bins], 
                         master_mean_dust_halo[model_number][snapshot_idx][nonzero_bins], 
                         label = label, ls = PlotScripts.linestyles[model_number], 
                         color = PlotScripts.colors[snapshot_idx]) 
                print(master_mean_dust_halo[model_number][snapshot_idx]) 
        for count, snapshot_idx in enumerate(PlotSnapList[model_number]):
            #label = r"$\mathbf{z = " + str(int(round(AllVars.SnapZ[snapshot_idx]))) + "}$"                
            label = r"$\mathbf{z = " + str(AllVars.SnapZ[snapshot_idx]) + "}$"                
            ax1.plot(np.nan, np.nan, ls = PlotScripts.linestyles[0], color =
                     PlotScripts.colors[count], label = label) 
            ax2.plot(np.nan, np.nan, ls = PlotScripts.linestyles[0], color =
                     PlotScripts.colors[count], label = label) 

        ax1.set_xlim([2.0, 10.5])
        #ax1.set_ylim([1.0, 6.0])

        ax1.set_xlabel(r'$\mathbf{log_{10} \: M_{*} \:[M_{\odot}]}$', fontsize = PlotScripts.global_labelsize) 
        ax1.set_ylabel(r'$\mathbf{log_{10} \: \langle M_{Dust}\rangle_{M*}}$', fontsize = PlotScripts.global_labelsize) 

        leg = ax1.legend(loc='upper left', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(PlotScripts.global_legendsize)

        outputFile1 = "./{0}_galaxy{1}".format(output_tag, output_format) 
        fig1.savefig(outputFile1, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile1))
        plt.close(fig1)
 
        ax2.set_xlim([6.8, 11.5])
        #ax2.set_ylim([1.0, 6.0])

        ax2.set_xlabel(r'$\mathbf{log_{10} \: M_{vir} \:[M_{\odot}]}$', fontsize = PlotScripts.global_labelsize) 
        ax2.set_ylabel(r'$\mathbf{log_{10} \: \langle M_{Dust}\rangle_{Mvir}}$', fontsize = PlotScripts.global_labelsize) 

        leg = ax2.legend(loc='upper left', numpoints=1, labelspacing=0.1)
        leg.draw_frame(False)  # Don't want a box frame
        for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize(PlotScripts.global_legendsize)

        outputFile2 = "./{0}_halo{1}".format(output_tag, output_format) 
        fig2.savefig(outputFile2, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile2))
        plt.close(fig2)


def plot_dust_scatter(SnapList, mass_gal, mass_halo, mass_dust, output_tag): 

    
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection='3d')

        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)

        ax1.scatter(mass_gal, mass_dust)
        ax2.scatter(mass_halo, mass_dust)
        #ax3.scatter(mass_gal, mass_halo, mass_dust)

        hb = ax4.hexbin(mass_halo, mass_dust, bins='log', cmap='inferno')
        ax1.set_xlabel(r'$\mathbf{log_{10} \: M_{*} \:[M_{\odot}]}$', fontsize = PlotScripts.global_labelsize) 
        ax1.set_ylabel(r'$\mathbf{log_{10} \: M_{Dust}}$', fontsize = PlotScripts.global_labelsize) 

        ax2.set_xlabel(r'$\mathbf{log_{10} \: M_{vir} \:[M_{\odot}]}$', fontsize = PlotScripts.global_labelsize) 
        ax2.set_ylabel(r'$\mathbf{log_{10} \: M_{Dust}}$', fontsize = PlotScripts.global_labelsize) 

        ax4.set_xlabel(r'$\mathbf{log_{10} \: M_{vir} \:[M_{\odot}]}$', fontsize = PlotScripts.global_labelsize) 
        ax4.set_ylabel(r'$\mathbf{log_{10} \: M_{Dust}}$', fontsize = PlotScripts.global_labelsize) 
        cb = fig4.colorbar(hb, ax=ax4)
        cb.set_label('log10(N)')

        outputFile1 = "./{0}_galaxy{1}".format(output_tag, output_format) 
        fig1.savefig(outputFile1, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile1))
        plt.close(fig1)

        outputFile2 = "./{0}_halo{1}".format(output_tag, output_format) 
        fig2.savefig(outputFile2, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile2))
        plt.close(fig2)

        #outputFile3 = "./{0}_3D{1}".format(output_tag, output_format) 
        #fig3.savefig(outputFile3, bbox_inches='tight')  # Save the figure
        #print('Saved file to {0}'.format(outputFile3))
        #plt.close(fig3)

        outputFile4 = "./{0}_hexbin{1}".format(output_tag, output_format) 
        fig4.savefig(outputFile4, bbox_inches='tight')  # Save the figure
        print('Saved file to {0}'.format(outputFile4))
        plt.close(fig4)

### Here ends the plotting functions. ###
### Here begins the functions that calculate various properties for the galaxies (fesc, Magnitude etc). ###

def Calculate_HaloPartStellarMass(halo_part, stellar_mass, bound_low, bound_high):
    '''
    Calculates the stellar mass for galaxies whose host halos contain a specified number of particles.

    Parameters
    ----------
    halo_part : array
        Array containing the number of particles inside each halo.
    stellar_mass : array
        Array containing the Stellar Mass for each galaxy (entries align with HaloPart). Units of log10(Msun).
    bound_low, bound_high : int
        We calculate the Stellar Mass of galaxies whose host halo has, bound_low <= halo_part <= bound_high.    

    Return
    -----
    mass, mass_std : float
        Mean and standard deviation stellar mass of galaxies whose host halo has number of particles between the specified bounds.  Units of log10(Msun)

    Units
    -----
    Input Stellar Mass is in units of log10(Msun).
    Output mean/std Stellar Mass is in units of log10(Msun).
    '''

    w = np.where((halo_part >= bound_low) & (halo_part <= bound_high))[0] # Find the halos with particle number between the bounds.

    mass = np.mean(10**(stellar_mass[w]))
    mass_std = np.std(10**(stellar_mass[w]))

    return np.log10(mass), np.log10(mass_std)

##


def calculate_fesc(fesc_prescription, fesc_normalization, halo_mass, ejected_fraction, quasar_fractional_boost):
    '''
    Calculate the escape fraction for a given prescription.

    Parameters
    ----------
    fesc_prescription : int
        Number that controls what escape fraction prescription we are using.
        0 : Constant, fesc = Constant.
        1 : Scaling with Halo Mass, fesc = A*Mh^B.
        2 : Scaling with ejected fraction, fesc = fej*A + B.
        3 : Depending on quasar activity, fesc = A (if no quasar within 1 dynamical time) or fesc = B (if quasar event within 1 dynamical time).
        4 : Anne's functional form for scaling positively with halo mass, fesc = B * (B / D)^(log(MH/A) / log(C/A)).
        5 : Anne's functional form for scaling inversely with halo mass, fesc = 1 - (1 - B) * (1 - B) / (1 - D) ^ (log(MH/A) / log(C/A))
    fesc_normalization : float (if fesc_prescription == 0) or `numpy.darray' with length 2 (if fesc_prescription == 1 or == 2).
        If fesc_prescription == 0, gives the constant value for the escape fraction.
        If fesc_prescription == 1 or == 2 or ==3, gives A and B with the form [A, B].
    halo_mass : `numpy.darray'
        Array that contains the halo masses for this snapshot. Units are log10(Msun).
    ejected_fraction : `numpy.darray'
        Array that contains the ejected fraction of the galaxies for this snapshot.
    quasar_fractional_boost : Array of floats with length equal to the number of galaxies.
        Array that contains the fraction of photons that should be boosted with the quasar activity.
        If the escape fraction should not be boosted by quasar activity, the index value with be 0.0. 
    
    Returns
    -------
    fesc : array, length is same as input arrays.
    Array containing the escape fraction for each galaxy for this snapshot.

    Units
    -----
    Input Halo Mass is in units of log10(Msun).
    '''

    if(len(halo_mass) != len(ejected_fraction)):
        print("The length of the halo_mass array is {0} and the length of the ejected_fraction array is {1}".format(len(halo_mass), len(ejected_fraction)))
        raise ValueError("These two should be equal.")

    if(fesc_prescription > 5):
        print("The prescription value you've chosen is {0}".format(fesc_prescription))
        raise ValueError("Currently we only have prescriptions 0 (constant), 1 (scaling with halo mass), 2 (scaling with ejected fraction) and 3 (boosted by  quasar activity).")

    if((fesc_prescription == 0 and isinstance(fesc_normalization, list) == True)):
        print("The escape fraction prescription is 0 but fesc_normalization was a list with value of {0}".format(fesc_normalization))
        raise ValueError("For a constant escape fraction, fesc_noramlization should be a single float.")

    if((fesc_prescription == 1 or fesc_prescription == 2 or fesc_prescription == 3 or fesc_prescription == 4 or fesc_prescription == 5) and (isinstance(fesc_normalization, list) == False)):
        print("The escape fraction prescription is {0} but fesc_normalization was not a list; it instead had a value of {1}".format(fesc_normalization))
        raise ValueError("For a scaling escape fraction fesc_normalization should be a list of the form [A, B]") 
 
    if (fesc_prescription == 0):
        fesc = np.full((len(halo_mass)), fesc_normalization)
    elif (fesc_prescription == 1):
        fesc = fesc_normalization[0]*pow(10,halo_mass*fesc_normalization[1])
    elif (fesc_prescription == 2):
        fesc = fesc_normalization[0]*ejected_fraction + fesc_normalization[1]
        fesc[fesc > 1.0] = 1.0
    elif (fesc_prescription == 3):                  
        fesc = fesc_normalization[0] * (1 - quasar_fractional_boost) + fesc_normalization[1] * quasar_fractional_boost # This is a tad subtle so I'll explain. 
        # quasar_fractional_boost is 0.0 if the escape fraction should not be boosted.  In this case, the escape fraction will be the base fraction.
        # If the quasar should boost the escape fraction for the entire snapshot, then quasar_fractional_boost = 1 and the escape fraction becomes the boosted fraction.
        # However if the quasar should boost the escape fraction for some fraction of the snapshot, then we perform the weighting and assign the escape fraction appropriately.
        # For example, consider a galaxy that emits 10^50 photons/s with a base escape fraction of 0.2 and boosted value of 1.0. 
        ## If we want three quarters of these photons to have the boosted escape fraction, then what should happen is (10^50 * (0.2 * 1/4) + 10^50 * (1.0 * 3/4)).
        ## In essence this is the exact same as assigning a single escape fraction of 0.2 * 1/4 + 1.0 * 3/4 = 0.8.
    elif (fesc_prescription == 4):  
        if (len(fesc_normalization) > 4):
            print("For escape fraction prescription {0}, there should be 4 constants specified.".format(fesc_prescription))

        Mh_low = fesc_normalization[0]        
        fesc_low = fesc_normalization[1]
        Mh_up = fesc_normalization[2]
        fesc_up = fesc_normalization[3]
        
        Mh = pow(10, halo_mass)        
        fesc = fesc_low * (fesc_low/fesc_up)**(-np.log10(Mh/Mh_low)/np.log10(Mh_up/Mh_low))

        fesc[fesc > fesc_low] = fesc_low

        #fesc = pow(fesc_normalization[1] * (fesc_normalization[1] / fesc_normalization[3]), -np.log10(pow(10,halo_mass) / fesc_normalization[0]) / np.log10(fesc_normalization[2] / fesc_normalization[0]))
    elif (fesc_prescription == 5):  
        if (len(fesc_normalization) > 4):
            print("For escape fraction prescription {0}, there should be 4 constants specified.".format(fesc_prescription))
        Mh_low = fesc_normalization[0]        
        fesc_low = fesc_normalization[1]
        Mh_up = fesc_normalization[2]
        fesc_up = fesc_normalization[3]
        
        Mh = pow(10, halo_mass)        

        fesc = 1. - (1.-fesc_low) * ((1.-fesc_low)/(1.-fesc_up))**(-np.log10(Mh/Mh_low)/np.log10(Mh_up/Mh_low))

        fesc[fesc < fesc_low] = fesc_low
        #fesc = 1 - pow((1 - fesc_normalization[1])  * ((1 - fesc_normalization[1]) / (1 - fesc_normalization[3])), -np.log10(pow(10, halo_mass) / fesc_normalization[0]) / np.log10(fesc_normalization[2] / fesc_normalization[0]))    

    if len(fesc) != 0:
    ## Adjust bad values, provided there isn't a riduculous amount of them. ##  
        w = np.where((halo_mass >= m_low) & (halo_mass <= m_high))[0] # Only care about the escape fraction values between these bounds. 
        nan_values = len(fesc[w][np.isnan(fesc[w])]) 
        aboveone_values = len(fesc[w][fesc[w] > 1.0])
        belowzero_values = len(fesc[w][fesc[w] < 0.0])
        bad_ratio =  (float(nan_values) + float(aboveone_values) + float(belowzero_values)) / float(len(fesc))
        
        if (bad_ratio > 0.10):
            print("The ratio of bad escape fractions to good is {0:.4f}".format(bad_ratio))
            print("Values above 1: {0}".format(fesc[fesc > 1.0]))
            w = np.where(fesc > 1.0)[0]
            print("Halo mass is: {0}".format(halo_mass[w]))
            print("Ejected fraction is: {0}".format(ejected_fraction[w]))
            
            print("Values below 0: {0}".format(fesc[fesc < 0.0]))
            w = np.where(fesc < 0.0)[0]
            print("Halo mass is: {0}".format(halo_mass[w]))
            print("Ejected fraction is: {0}".format(ejected_fraction[w]))
            if (len(fesc) > 100): # Only throw an error if we had an appreciable number of bad fesc.
                raise ValueError("This was above the tolerance level of 10%.")

        fesc[np.isnan(fesc)] = 0 # Get rid of any lingering Nans.
        fesc[fesc > 1] = 1.0  # Get rid of any values above 1. 
        fesc[fesc < 0] = 0.0  # Get rid of any values below 0.
    else:
        fesc = np.nan   
     
    return fesc

##
def calculate_photons(SFR, Z):
    '''
    This calculates the number of HI ionizing photons emitted by a galaxy of a given star formation rate and metallicity.
    Fit is based on the Stellar Synthesis of STARBURST99 (Leither et al., 1999).

    Parameters
    ----------
    SFR : `numpy.darray' with length equal to the number of galaxies.
    Star formation rate for each galaxy. Units of log10(Msun yr^-1).
    Z : `numpy.darray' with length equal to the number of galaxies.
    Metallicity (pure, not solar) for each galaxy.

    Returns
    -------
    ngamma_HI : `numpy.darray' with length equal to the number of galaxies.
    Number of HI ionizing photons emitted for each galaxy. Units of log10(s^-1).

    Units
    -----
    Star Formation Rate is in units of log10(Msun yr^-1).
    Metallicity is in proper metallicity (not solar).
    ngamma_HI is in units of log10(s^-1).
    '''

    ngamma_HI = []

    ## Fits are based on the blah tracks of STARBURST99. ##

    for i in range(0, len(SFR)):
        ngamma_HI_tmp = 0.0
        if (SFR[i] == 0.0):
            n_gamma_HI_tmp = 0
        elif (Z[i] < 0.0025):
            ngamma_HI_tmp = SFR[i] + 53.354
        elif (Z[i] >= 0.0025 and Z[i] < 0.006):
            ngamma_HI_tmp = SFR[i] + 53.290
        elif (Z[i] >= 0.006 and Z[i] < 0.014):
            ngamma_HI_tmp = SFR[i] + 53.248
        elif (Z[i] >= 0.014 and Z[i] < 0.30):
            ngamma_HI_tmp = SFR[i] + 53.166
        else:
            ngamma_HI_tmp = SFR[i] + 53.041  
        
        ngamma_HI.append(ngamma_HI_tmp.astype(np.float32))
    return ngamma_HI

##
def calculate_UV_extinction(z, L, M):
    '''
    Calculates the observed UV magnitude after dust extinction is accounted for.

    Parameters
    ----------
    z : float
    Redshift we are calculating the extinction at.
    L, M : array, length equal to the number of galaxies at this snapshot.
    Array containing the UV luminosities and magnitudes.

    Returns
    -------
    M_UV_obs : array, length equal to the number of galaxies at this snapshot.
    Array containing the observed UV magnitudes.

    Units
    -----
    Luminosities are in units of log10(erg s^-1 A^-1).
    Magnitudes are in the AB system.
    '''

    M_UV_bins = np.arange(-24, -16, 0.1)
    A_mean = np.zeros((len(MUV_bins))) # A_mean is the average UV extinction for a given UV bin.    

    for j in range(0, len(M_UV_bins)):
        beta = calculate_beta(M_UV_bins[j], AllVars.SnapZ[current_snap]) # Fits the beta parameter for the current redshift/UV bin. 
        dist = np.random.normal(beta, 0.34, 10000) # Generates a normal distribution with mean beta and standard deviation of 0.34.
        A = 4.43 + 1.99*dist 
        A[A < 0] = 0 # Negative extinctions don't make sense.
            
        A_Mean[j] = np.mean(A)

    indices = np.digitize(M, M_UV_bins) # Bins the simulation magnitude into the MUV bins. Note that digitize defines an index i if bin[i-1] <= x < bin[i] whereas I prefer bin[i] <= x < bin[i+1]
    dust = A_Mean[indices]
    flux = AllVars.Luminosity_to_Flux(L, 10.0) # Calculate the flux from a distance of 10 parsec, units of log10(erg s^-1 A^-1 cm^-2). 
    flux_observed = flux - 0.4*dust
    
    f_nu = ALlVars.spectralflux_wavelength_to_frequency(10**flux_observed, 1600) # Spectral flux desnity in Janksy.
    M_UV_obs(-2.5 * np.log10(f_nu) + 8.90) # AB Magnitude from http://www.astro.ljmu.ac.uk/~ikb/convert-units/node2.html

    return M_UV_obs

##

def update_cumulative_stats(mean_pool, std_pool, N_pool, mean_local, std_local, N_local):
    '''
    Update the cumulative statistics (such as Stellar Mass Function, Mvir-Ngamma, fesc-z) that are saved across files.
    Pooled mean formulae taken : from https://www.ncbi.nlm.nih.gov/books/NBK56512/
    Pooled variance formulae taken from : https://en.wikipedia.org/wiki/Pooled_variance

    Parameters
    ----------
    mean_pool, std_pool, N_pool : array of floats with length equal to the number of bins (e.g. the mass bins for the Stellar Mass Function).
        The current mean, standard deviation and number of data points within in each bin.  This is the array that will be updated in this function.
    mean_local, std_local, N_local : array of floats with length equal to the number of bins.
        The mean, standard deviation and number of data points within in each bin that will be added to the pool.

    Returns
    -------
    mean_pool, std_pool, N_pool : (See above)
    The updated arrays with the local values added and accounted for within the pools.

    Units
    -----
    All units are kept the same as the input units.
    Values are in real-space (not log-space).
    '''
   
    N_times_mean_local = np.multiply(N_local, mean_local)
    N_times_var_local = np.multiply(N_local - 1, np.multiply(std_local, std_local)) # Actually N - 1 because of Bessel's Correction 
                                        # https://en.wikipedia.org/wiki/Bessel%27s_correction).  #
    N_times_mean_pool = np.add(N_times_mean_local, np.multiply(N_pool, mean_pool))
    N_times_var_pool = np.add(N_times_var_local, np.multiply(N_pool - 1, np.multiply(std_pool, std_pool)))
    N_pool = np.add(N_local, N_pool)

    '''
    print(mean_local)
    print(type(mean_local))
    print((type(mean_local).__module__ == np.__name__))
    print(isinstance(mean_local, list))
    print(isinstance(mean_local,float64))
    print(isinstance(mean_local,float32))
    '''
    if (((type(mean_local).__module__ == np.__name__) == True or (isinstance(mean_local, list) == True)) and isinstance(mean_local, float) == False and isinstance(mean_local, int) == False and isinstance(mean_local,float32) == False and isinstance(mean_local, float64) == False): # Checks to see if we are dealing with arrays. 
        for i in range(0, len(N_pool)):
            if(N_pool[i] == 0): # This case is when we have no data points in the bin. 
                mean_pool[i] = 0.0
            else:
                mean_pool[i] = N_times_mean_pool[i]/N_pool[i]
            if(N_pool[i] < 3): # In this instance we don't have enough data points to properly calculate the standard deviation.
                std_pool[i] = 0.0
            else:
                std_pool[i] = np.sqrt(N_times_var_pool[i]/ (N_pool[i] - 2)) # We have -2 because there is two instances of N_pool contains two 'N - 1' terms. 
        
    else:
        mean_pool = N_times_mean_pool / N_pool

        if(N_pool < 3):
            std_pool = 0.0
        else:
            std_pool = np.sqrt(N_times_var_pool / (N_pool - 2))
 
    return mean_pool, std_pool

    ### Here ends the functions that deal with galaxy data manipulation. ###

def do_quasar_tracking(quasar_tracking, N_dyntime, NumSubsteps, current_snap, w_gal, gal_DynamicalTime, gal_QuasarActivity, gal_QuasarSubstep):
    '''
    Update the quasar tracking arrays that decide when/if the escape fraction of a galaxy should be boosted due to recent quasar activity.
    The passed arrays will contain information for an entire file but only calculated for a single snapshot (done to preserve indexing). 

    Parameters
    ----------
    quasar_tracking : Structured array with data-type defined by the 'quasar_tracking_dtype' variable. 
        Contains all the arrays for tracking the quasar activity, quasar boost fraction etc for an entire file. 
    N_dyntime : float
        The number of dynamical times after a quasar event that the galaxy returns to its fiducial value.    
    NumSubsteps : float
        The number of substeps for this model.  Used to determine the fraction of the snapshot size the quasar went off.
    current_snap : int
        Current snapshot we are updating the tracking for.
    w_gal : array of integers length equal to the number of 'good' galaxies for this snapshot.
        Indices of galaxies that are to be used for this snapshot. These galaxies exist at the current_snap, have the acceptable amount of halo particles etc.
    gal_DynamicalTime : array of floats with length equal to the number of galaxies in the file.
        The dynamical time of the host halo for all galaxies in this file.
    gal_QuasarActivity, gal_QuasarSubstep : Nested arrays of integers with outer length equal to the number of galaxies in the file and inner length equal to the number of snapshot.
        Contains the data for all galaxies in the file for whether a quasar has gone off at the specified snapshot, and if so, what substep it won't off during.

    Returns
    -------
    quasar_tracking : (See above)
        Updated quasar tracking arrays.
    N_quasars_tmp, N_quasars_boost_tmp : floats
        Number of quasars that went off during this snapshot and number of galaxies that are having their escape fraction boosted during this snapshot.
    dynamicaltime_quasars_tmp : array of floats with length equal to the number of quasars that went off this snapshot.
        Dynamical time of halos in which a quasar went off. 

    Units
    -----
    All units are kept the same as the input units.
    Times (dynamical time, boost time etc) are all in Myr. 
    '''
                    
    N_quasars_tmp = 0.0
    N_quasars_boost_tmp = 0.0
    dynamicaltime_quasars_tmp = [] 
    
    ## Check to see if any quasars went off during this snapshot. ## 
    w_quasar_activity = w_gal[np.where((gal_QuasarActivity[w_gal, current_snap] == 1))[0]] 
    if (len(w_quasar_activity) > 0): # If a quasar went off...
        N_quasars_tmp += len(w_quasar_activity)                   
        quasar_tracking['QuasarActivityToggle'][w_quasar_activity] += 1 # Turn on the boosted escape fraction.
        quasar_tracking['QuasarSnapshot'][w_quasar_activity] = current_snap # Remember the snapshot that this quasar went off.
        quasar_tracking['TargetQuasarTime'][w_quasar_activity] = gal_DynamicalTime[w_quasar_activity, current_snap] * N_dyntime # Keep track of how long we want the boosted escape fraction to last for.
        quasar_tracking['QuasarActivitySubstep'][w_quasar_activity] = gal_QuasarSubstep[w_quasar_activity, current_snap] # The substep of the snapshot the quasar went off. 
        dynamicaltime_quasars_tmp.extend(gal_DynamicalTime[w_quasar_activity, current_snap])
        quasar_tracking['QuasarBoostActiveTime'][w_quasar_activity] = 0.0;

    # We need to handle the case in which a quasar boosted galaxy turned off last snapshot.  
    # In this case, we just need to reset the fractional photon boosting.

    w_turned_off_lastsnap = w_gal[np.where((quasar_tracking['QuasarActivityToggle'][w_gal] < 1.0))[0]]
    if (len(w_turned_off_lastsnap) > 0):
        quasar_tracking['QuasarFractionalPhoton'][w_turned_off_lastsnap] = 0.0

    ## Update the boost times and see if we need to turn anything off ## 
    w_active_boost = w_gal[np.where((quasar_tracking['QuasarActivityToggle'][w_gal] > 0.0))[0]] # Use greater than because comparing a float.
    if (len(w_active_boost) > 0):
        quasar_snap = quasar_tracking['QuasarSnapshot'][w_active_boost]
        N_quasars_boost_tmp += len(w_active_boost)
        
        assert(quasar_snap.any() != -1) # Just a sanity check.

        dt = ((AllVars.Lookback_Time[current_snap - 1]) - (AllVars.Lookback_Time[current_snap])) * 1.0e3 # Time between the previous snapshot and now (in Myr). 
      
        w_just_turned_on = w_active_boost[np.where((quasar_snap == current_snap) & (quasar_tracking['QuasarActivityToggle'][w_active_boost] > 0.5) & (quasar_tracking['QuasarActivityToggle'][w_active_boost] < 1.5))[0]] # In this case, the boosted escape fraction turned on during this snapshot.  As a result, we need to account for it turning on during a substep and only providing a fractional boost.
        # Note, the check that QuasarActivityToggle is 1 (between 0.5 and 1.5) as if we have the case where a quasar went off while a galaxy was still being boosted, we don't want to use a fractional boost.
        # While yes, there could be an instance where the boosting is set to turn off by substep 1, and then the second quasar doesn't go off until a later substep, the cases where this would happen is small and the overall effect neglible. 
        w_not_just_turned_on = w_active_boost[np.where((quasar_snap != current_snap))[0]] 

        quasar_tracking['QuasarBoostActiveTime'][w_just_turned_on] += dt * (NumSubsteps - quasar_tracking['QuasarActivitySubstep'][w_just_turned_on]) / NumSubsteps # This adds the time spanned by this snapshot weighted by when substep in which the quasar went off. E.g., if there are 10 substeps and the quasar went off during substep 3, then the boosted escape fraction occurs for (10 - 3) / 10 = 0.7 of the time.  
        quasar_tracking['QuasarFractionalPhoton'][w_just_turned_on] = (NumSubsteps - quasar_tracking['QuasarActivitySubstep'][w_just_turned_on]) / NumSubsteps # Only boost the photons for a fraction of the time during the snapshot step. 

        quasar_tracking['QuasarBoostActiveTime'][w_not_just_turned_on] += dt # The boosted escape fraction lasts for the entire snapshot.
        quasar_tracking['QuasarFractionalPhoton'][w_not_just_turned_on] = 1.0

        w_shutoff = w_active_boost[np.greater(quasar_tracking['QuasarBoostActiveTime'][w_active_boost], quasar_tracking['TargetQuasarTime'][w_active_boost]).nonzero()] # If it's been longer than N dynamical times, turn the boosted escape fraction off.
# np.greater returns True if it's time to turn off, .nonzero() returns the indices of those, and then we want the index within the GLOBAL galaxy array so wrap it all within w_active_boost.

        # Now for those galaxies for which we wish to shut off their boosted escape fraction, we need to determine on which substep we are turning it off.
        # This will then give us a fraction of the photons emitted that we should boost for this final time.

        TimeIntoSnapshot = quasar_tracking['TargetQuasarTime'][w_shutoff] - (quasar_tracking['QuasarBoostActiveTime'][w_shutoff] - dt) # This tells us how much time into this snapshot this quasar will be turned on for.
        FractionIntoSnapshot = TimeIntoSnapshot / dt # Then this is the fraction of time the quasar will be on for.
        quasar_tracking['QuasarFractionalPhoton'][w_shutoff] = FractionIntoSnapshot # Weight the boosted escape fraction to be correct. E.g., if a galaxy is turned off 40% into the snapshot, then the escape fraction should only be boosted for 40% of the time.
               
        quasar_tracking['QuasarActivityToggle'][w_shutoff] -= 1.0 # Then turn off all the toggles.
        quasar_tracking['QuasarSnapshot'][w_shutoff] = 0
        quasar_tracking['TargetQuasarTime'][w_shutoff] = 0.0
        quasar_tracking['QuasarBoostActiveTime'][w_shutoff] = 0.0
        quasar_tracking['QuasarActivitySubstep'][w_shutoff] = -1

    return quasar_tracking, N_quasars_tmp, N_quasars_boost_tmp, dynamicaltime_quasars_tmp     

def determine_MH_fesc_constants(low_MH, low_fesc, high_MH, high_fesc):
    
    log_A = (np.log10(high_fesc) - (np.log10(low_fesc)*np.log10(high_MH)/np.log10(low_MH))) * pow(1 - (np.log10(high_MH) / np.log10(low_MH)), -1)
    B = (np.log10(low_fesc) - log_A) / np.log10(low_MH)
    A = pow(10, log_A)

    return A, B

#################################

if __name__ == '__main__':
    
    print("This code runs in either mode 0 or mode 1")
    print("Mode 0 runs in the old way by going into the Python script and editing all the arrays manually.")
    print("Mode 1 runs specifically for the goodness-of-fit analysis using the quasar fesc prescription. This mode still need to adjust all the arrays EXCEPT the fesc_prescription/normalization.")
    if (len(sys.argv) < 2):
        print("Usage: python3 myresults.py <Mode>")        
        exit()

    mode = int(sys.argv[1])
    if (mode < 0 or mode > 1):
        print("Mode of operation should be either 0 (default run, go into script and modify params) or 1 (quasar goodness-of-fit analysis)")
        exit()

    if (len(sys.argv) != 5 and mode == 1):
        print("It looks like you're trying to use the Quasar goodness-of-fit analysis.")
        print("If mode of operation == 1...") 
        print("Usage: python3 myresults.py <Mode> <Baseline fesc> <Boosted fesc> <Number of Dynamical Times to be boosted for>") 
        exit()


    if (mode == 1): 
        baseline_fesc = float(sys.argv[2])
        boosted_fesc = float(sys.argv[3])
        boosted_dynamicaltime = float(sys.argv[4])
   
    np.seterr(divide='ignore')
    number_models = 2

    galaxies_model1 = '/lustre/projects/p004_swin/jseiler/kali/self_consistent_GridReionMod/galaxies/base_z5.782'
    merged_galaxies_model1 = '/lustre/projects/p004_swin/jseiler/kali/self_consistent_GridReionMod/galaxies/base_MergedGalaxies'

    galaxies_model2 = '/lustre/projects/p004_swin/jseiler/kali/self_consistent_use_analytic/galaxies/base_z5.782'
    merged_galaxies_model2 = '/lustre/projects/p004_swin/jseiler/kali/self_consistent_use_analytic/galaxies/base_MergedGalaxies'

    galaxies_model3 = '/lustre/projects/p004_swin/jseiler/kali/base_reionization_on/galaxies/base_z5.782'
    merged_galaxies_model3 = '/lustre/projects/p004_swin/jseiler/kali/base_reionization_on/galaxies/base_MergedGalaxies'

    galaxies_model4='/lustre/projects/p004_swin/jseiler/kali/dust/galaxies/dust_z5.782'
    merged_galaxies_model4='/lustre/projects/p004_swin/jseiler/kali/dust/galaxies/dust_MergedGalaxies'

    galaxies_model5 ='/lustre/projects/p004_swin/jseiler/kali/lowquasar/galaxies/lowquasar_z5.782'
    merged_galaxies_model5 ='/lustre/projects/p004_swin/jseiler/kali/lowquasar/galaxies/lowquasar_MergedGalaxies'

    galaxies_model6='/lustre/projects/p004_swin/jseiler/kali/IRA/galaxies/IRA_z5.782'
    merged_galaxies_model6='/lustre/projects/p004_swin/jseiler/kali/IRA/galaxies/IRA_MergedGalaxies'

    galaxies_model7='/lustre/projects/p004_swin/jseiler/mini_millennium/dust/galaxies/dust_z0.000'
    merged_galaxies_model7='/lustre/projects/p004_swin/jseiler/mini_millennium/dust/galaxies/dust_MergedGalaxies'

    galaxies_model8='/lustre/projects/p004_swin/jseiler/kali/tmp/galaxies/oldBH_z5.782'
    merged_galaxies_model8='/lustre/projects/p004_swin/jseiler/kali/tmp/galaxies/oldBH_MergedGalaxies'
        
    galaxies_model9='/lustre/projects/p004_swin/jseiler/kali/tmp/galaxies/newBH_z5.782'
    merged_galaxies_model9='/lustre/projects/p004_swin/jseiler/kali/tmp/galaxies/newBH_MergedGalaxies'

    galaxies_filepath_array = [galaxies_model8, galaxies_model9]
    merged_galaxies_filepath_array = [merged_galaxies_model8, 
                                      merged_galaxies_model9]
       
    number_substeps = [10, 10] # How many substeps does each model have (specified by STEPS variable within SAGE).
    number_snapshots = [99, 99] # Number of snapshots in the simulation (we don't have to do calculations for ALL snapshots).
    # Tiamat extended has 164 snapshots.
     
    FirstFile = [0, 0] # The first file number THAT WE ARE PLOTTING.
    LastFile = [63, 63] # The last file number THAT WE ARE PLOTTING.
    NumFile = [64, 64] # The number of files for this simulation (plotting a subset of these files is allowed).     
    same_files = [0, 0] # In the case that model 1 and model 2 (index 0 and 1) have the same files, we don't want to read them in a second time.
    # This array will tell us if we should keep the files for the next model or otherwise throw them away. 
    # The files will be kept until same_files[current_model_number] = 0.
    # For example if we had 5 models we were plotting and model 1, 2, 3 shared the same files and models 4, 5 shared different files,
    # Then same_files = [1, 1, 0, 1, 0] would be the correct values.

    done_model = np.zeros((number_models)) # We use this to keep track of if we have done a model already.
    model_tags = [r"$\mathrm{OldBH}$",
                  r"$\mathrm{NewBH}$"]

    ## Constants used for each model. ##
    # Need to add an entry for EACH model. #

    halo_cut = [32, 32, 32] # Only calculate properties for galaxies whose host halos have at least this many particles.

    ### fesc Stuff ###
    fesc_prescription = [0, 0] # This defines what escape fractions prescription we want to use for each mode. 
    # 0 is constant.
    # 1 is scaling with halo mass. 
    # 2 is scaling with ejected fraction. 
    # 3 is a boosted escape fraction depending upon quasar activity. 
    # 4 is Anne's Functional form that scales inversely with halo mass (smaller fesc for higher halo mass).
    # 5 is Anne's function form that scales with halo mass (larger fesc for higher halo mass).

    #fesc_normalization = [[0.2, 1.0, 2.5], [0.2, 1.0, 2.5]] # Normalization constants for each escape fraction prescription. The value depends upon the prescription selected.
    fesc_normalization = [0.3, 0.3] # Normalization constants for each escape fraction prescription. The value depends upon the prescription selected.
    # For prescription 0, requires a number that defines the constant fesc.
    # For prescription 1, fesc = A*M^B. Requires an array with 2 numbers the first being A and the second B.
    # For prescription 2, fesc = A*fej + B.  Requires an array with 2 numbers the first being A and the second B.
    # For prescription 3, fesc = A (if no quasar within 1 dynamical time) or fesc = B (if quasar event within C dynamical times). Requires an array with 2 numbers the first being A, the second B and the third C.
    # For prescription 4, fesc = B * (B / D)^(log(MH/A) / log(C/A)).
    # For prescription 5, fesc = 1 - (1 - B) * (1 - B) / (1 - D) ^ (log(MH/A) / log(C/A))
   
    # For Tiamat, z = [6, 7, 8] are snapshots [78, 64, 51]
    # For Kali, z = [6, 7, 8] are snapshots [93, 76, 64]
    #SnapList = [np.arange(0,99), np.arange(0,99)] # These are the snapshots over which the properties are calculated. NOTE: If the escape fraction is selected (fesc_prescription == 3) then this should be ALL the snapshots in the simulation as this prescriptions is temporally important. 
    #SnapList = [np.arange(20,99), np.arange(20, 99), np.arange(20, 99)]    
    #SnapList = [[30, 50, 64, 76, 93]]
    SnapList = [[93, 76, 64],
                [93, 76, 64]]
    #PlotSnapList = [[93, 76, 64]]
    PlotSnapList = SnapList 

    simulation_norm = [5, 5] # Changes the constants (cosmology, snapshot -> redshift mapping etc) for each simulation. 
    # 0 for MySim (Manodeep's old one).
    # 1 for Mini-Millennium.
    # 2 for Tiamat (up to z =5).
    # 3 for extended Tiamat (down to z = 1.6ish). 
    # 4 for Britton's Sim Pip
    # 5 for Manodeep's new simulation Kali.

    stellar_mass_halolen_lower = [32, 95, 95, 95] # These limits are for the number of particles in a halo.  
    stellar_mass_halolen_upper = [50, 105, 105, 105] # We calculate the average stellar mass for galaxies whose host halos have particle count between these limits.
    calculate_observed_LF = [0, 0, 0] # Determines whether we want to account for dust extinction when calculating the luminosity function of each model.

    paper_plots = 1 

    ## If we are running in Mode 1, we want to specifically do the quasar prescription with the constants defined by the inputs ##
    if (mode == 1):
        if (number_models != 1):
            print("The number of models can only be 1 if running in Mode 1.")
            exit()
        fesc_prescription = [3]
        fesc_normalization = [[baseline_fesc, boosted_fesc, boosted_dynamicaltime]]        
    
    ##############################################################################################################
    ## Do a few checks to ensure all the arrays were specified properly. ##

    files = [galaxies_filepath_array, merged_galaxies_filepath_array, number_snapshots, model_tags, SnapList]
    goodness_check = [len(x) == len(y) for i,x in enumerate(files) for j,y in enumerate(files) if i != j] # This goes through the files array and checks to see if all the files have the same lengths.

    if False in goodness_check: # If any of the arrays had different lengths, throw an error.
        print("One of the input arrays had an incorrect length")
        exit() 

    if number_models != len(galaxies_filepath_array):
        print("The number of models was given as {0} whereas we had {1} filepaths given.".format(number_models, len(galaxies_filepath_array)))
        exit()

    for model_number in range(0,number_models):
        assert(LastFile[model_number] - FirstFile[model_number] + 1 >= size)

        if(simulation_norm[model_number] == 1):
            AllVars.Set_Params_MiniMill()
        elif(simulation_norm[model_number] == 3):
            AllVars.Set_Params_Tiamat_extended()
        elif(simulation_norm[model_number] == 4):
            AllVars.Set_Params_Britton()
        elif(simulation_norm[model_number] == 5):
            AllVars.Set_Params_Kali()
        else: 
            print("Simulation norm was set to {0}.".format(simulation_norm[model_number]))
            raise ValueError("This option has been implemented yet.  Get your head in the game Jacob!")

        if (number_snapshots[model_number] != len(AllVars.SnapZ)): # Here we do a check to ensure that the simulation we've defined correctly matches the number of snapshots we have also defined. 
            print("The number_snapshots array is {0}".format(number_snapshots))
            print("The simulation_norm array is {0}".format(simulation_norm))
            print("The number of snapshots for model_number {0} has {1} but you've said there is only {2}".format(model_number, len(AllVars.SnapZ), number_snapshots[model_number]))
            raise ValueError("Check either that the number of snapshots has been defined properly and that the normalization option is correct.")


    ######################################################################   
    ##################### SETTING UP ARRAYS ##############################
    ######################################################################   
    
    ### The arrays are set up in a 3 part process. ###
    ### This is because our arrays are 3D nested to account for the model number and snapshots. ### 

    # First set up the outer most array. #

    ## Arrays for functions of stellar mass. ##
    SMF = [] # Stellar Mass Function.
    mean_fesc_galaxy_array = [] # Mean escape fraction as a function of stellar mass.
    std_fesc_galaxy_array = [] # Same as above but standard devation.
    N_galaxy_array = [] # Number of galaxies as a function of stellar mass.
    mean_BHmass_galaxy_array = [] # Black hole mass as a function of stellar mass.
    std_BHmass_galaxy_array = [] # Same as above but standard deviation. 
    mergers_galaxy_array = [] # Number of mergers as a function of halo mass. 

    mean_dust_galaxy_array = [] # Mean dust mass as a function of stellar mass. 
    std_dust_galaxy_array = [] # Same as above but standard deviation. 

    ## Arrays for functions of halo mass. ##
    mean_ejected_halo_array = [] # Mean ejected fractions as a function of halo mass.
    std_ejected_halo_array = [] # Same as above but standard deviation.
    mean_fesc_halo_array = [] # Mean escape fraction as a function of halo mass.
    std_fesc_halo_array = [] # Same as above but standard deviation.
    mean_Ngamma_halo_array = [] # Mean number of ionizing photons THAT ESCAPE as a function of halo mass.
    std_Ngamma_halo_array = [] # Same as above but standard deviation.
    N_halo_array = [] # Number of galaxies as a function of halo mass.

    mergers_halo_array = [] # Number of mergers as a function of halo mass. 

    mean_quasar_activity_array = [] # Mean fraction of galaxies that have quasar actvitity as a function of halo mas.
    std_quasar_activity_array = [] # Same as above but standard deviation.

    mean_reionmod_halo_array = [] # Mean reionization modifier as a function of halo mass.
    std_reionmod_halo_array = [] # Same as above but for standard deviation. 

    ## Arrays for functions of redshift. ##
    sum_Ngamma_z_array = [] # Total number of ionizing photons THAT ESCAPE as a functio of redshift. 
    mean_fesc_z_array = [] # Mean number of ionizing photons THAT ESCAPE as a function of redshift.
    std_fesc_z_array = [] # Same as above but standard deviation.
    N_z = [] # Number of galaxies as a function of redshift. 
    galaxy_halo_mass_mean = [] # Mean galaxy mass as a function of redshift.

    N_quasars_z = [] # This tracks how many quasars went off during a specified snapshot.
    N_quasars_boost_z = [] # This tracks how many galaxies are having their escape fraction boosted by quasar activity.

    dynamicaltime_quasars_mean_z = [] # Mean dynamical time of galaxies that have a quasar event as a function of redshift.
    dynamicaltime_quasars_std_z = [] # Same as above but standard deviation.
    dynamicaltime_all_mean_z = [] # Mean dynamical time of all galaxies.
    dynamicaltime_all_std_z = [] # Same as above but standard deviation.

    mean_reionmod_z = [] # Mean reionization modifier as a function of redshift.
    std_reionmod_z = [] # Same as above but for standard deviation. 
    N_reionmod_z = [] # Number of galaxies with a non-negative reionization modifier.

    mean_dust_halo_array = [] # Mean dust mass as a function of stellar mass. 
    std_dust_halo_array = [] # Same as above but standard deviation. 

    ## Now the outer arrays have been defined, set up the next nest level for the number of models. ##

    for model_number in range(0,number_models):
        ## Galaxy Arrays ##
        SMF.append([])
        mean_fesc_galaxy_array.append([])
        std_fesc_galaxy_array.append([])
        N_galaxy_array.append([])
        mean_BHmass_galaxy_array.append([])
        std_BHmass_galaxy_array.append([])
        mergers_galaxy_array.append([]) 

        mean_dust_galaxy_array.append([])
        std_dust_galaxy_array.append([])

        ## Halo arrays. ##
        mean_ejected_halo_array.append([])
        std_ejected_halo_array.append([])
        mean_fesc_halo_array.append([])
        std_fesc_halo_array.append([])
        mean_Ngamma_halo_array.append([])
        std_Ngamma_halo_array.append([])
        N_halo_array.append([])

        mergers_halo_array.append([]) 

        mean_quasar_activity_array.append([])
        std_quasar_activity_array.append([])

        mean_reionmod_halo_array.append([])
        std_reionmod_halo_array.append([])

        mean_dust_halo_array.append([])
        std_dust_halo_array.append([])

        ## Redshift arrays. ##
        sum_Ngamma_z_array.append([])
        mean_fesc_z_array.append([])
        std_fesc_z_array.append([])
        N_z.append([])
        galaxy_halo_mass_mean.append([])

        N_quasars_z.append([])
        N_quasars_boost_z.append([])

        dynamicaltime_quasars_mean_z.append([])
        dynamicaltime_quasars_std_z.append([])
        dynamicaltime_all_mean_z.append([])
        dynamicaltime_all_std_z.append([])

        mean_reionmod_z.append([])
        std_reionmod_z.append([])
        N_reionmod_z.append([])

        ## And then finally set up the inner most arrays ##
        ## NOTE: We do the counts as float so we can keep consistency when we're calling MPI operations (just use MPI.FLOAT rather than deciding if we need to use MPI.INT)

        for snapshot_idx in range(len(SnapList[model_number])):
            ## For the arrays that are functions of stellar/halo mass, the inner most level will be an array with the statistic binned across mass ##
            ## E.g. SMF[model_number][snapshot_idx] will return an array whereas N_z[model_number][snapshot_idx] will return a float. ##
 
            ## Functions of stellar mass arrays. ##
            SMF[model_number].append(np.zeros((NB_gal), dtype = np.float32)) 
            mean_fesc_galaxy_array[model_number].append(np.zeros((NB_gal), dtype = np.float32)) 
            std_fesc_galaxy_array[model_number].append(np.zeros((NB_gal), dtype = np.float32))
            N_galaxy_array[model_number].append(np.zeros((NB_gal), dtype = np.float32)) 
            mean_BHmass_galaxy_array[model_number].append(np.zeros((NB_gal), dtype = np.float32)) 
            std_BHmass_galaxy_array[model_number].append(np.zeros((NB_gal), dtype = np.float32)) 
            mergers_galaxy_array[model_number].append(np.zeros((NB_gal), dtype = np.float32))

            mean_dust_galaxy_array[model_number].append(np.zeros((NB_gal), dtype = np.float32)) 
            std_dust_galaxy_array[model_number].append(np.zeros((NB_gal), dtype = np.float32)) 

            ## Function of halo mass arrays. ##
            mean_ejected_halo_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
            std_ejected_halo_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
            mean_fesc_halo_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
            std_fesc_halo_array[model_number].append(np.zeros((NB), dtype = np.float32))
            mean_Ngamma_halo_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
            std_Ngamma_halo_array[model_number].append(np.zeros((NB), dtype = np.float32))
            N_halo_array[model_number].append(np.zeros((NB), dtype = np.float32))

            mergers_halo_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
 
            mean_quasar_activity_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
            std_quasar_activity_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
    
            mean_reionmod_halo_array[model_number].append(np.zeros((NB), dtype = np.float32))
            std_reionmod_halo_array[model_number].append(np.zeros((NB), dtype = np.float32))

            mean_dust_halo_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
            std_dust_halo_array[model_number].append(np.zeros((NB), dtype = np.float32)) 
 
            ## Function of Redshift arrays. ##
            sum_Ngamma_z_array[model_number].append(0.0) 
            mean_fesc_z_array[model_number].append(0.0) 
            std_fesc_z_array[model_number].append(0.0)
            N_z[model_number].append(0.0) 
            galaxy_halo_mass_mean[model_number].append(0.0)

            N_quasars_z[model_number].append(0.0) 
            N_quasars_boost_z[model_number].append(0.0)

            dynamicaltime_quasars_mean_z[model_number].append(0.0)
            dynamicaltime_quasars_std_z[model_number].append(0.0)
            dynamicaltime_all_mean_z[model_number].append(0.0)
            dynamicaltime_all_std_z[model_number].append(0.0)

            mean_reionmod_z[model_number].append(0.0)
            std_reionmod_z[model_number].append(0.0)
            N_reionmod_z[model_number].append(0.0)

    ## Define structured arrays ##

    Quasar_Tracking_full = [
    ('QuasarActivityToggle',    np.float32), # Array to specify whether the galaxy should have the baseline or boosted escape fraction.
    ('QuasarActivitySubstep',   np.int32), # Array to specify in which substep the quasar boosting begins. 
    ('QuasarSnapshot',          np.int32), # Array to keep track of when the quasar went off. 
    ('TargetQuasarTime',        np.float32), # Array to keep track of the amount of time we want to keep the boosted escape fraction for. 
    ('QuasarBoostActiveTime',   np.float32), # Array that tracks how long it has been since the quasar boosting was turned on.
    ('QuasarFractionalPhoton',  np.float32) # Array to keep track of what fraction of photons we should boost with the prescription.             
                     ]

    names = [Quasar_Tracking_full[i][0] for i in range(len(Quasar_Tracking_full))]
    formats = [Quasar_Tracking_full[i][1] for i in range(len(Quasar_Tracking_full))]
    quasar_tracking_dtype = np.dtype({'names':names, 'formats':formats}, align=True)

    ######################################################################   
    #################### ALL ARRAYS SETUP ################################
    ######################################################################   

    ## Now it's (finally) time to read in all the data and do the actual work. ##

    for model_number in range(number_models):

        if(simulation_norm[model_number] == 1):
            AllVars.Set_Params_MiniMill()            
        elif(simulation_norm[model_number] == 3):
            AllVars.Set_Params_Tiamat_extended()
        elif(simulation_norm[model_number] == 4):
            AllVars.Set_Params_Britton()
        elif(simulation_norm[model_number] == 5):
            AllVars.Set_Params_Kali()
        else: 
            print("Simulation norm was set to {0}.".format(simulation_norm[model_number]))
            raise ValueError("This option has been implemented yet.  Get your head in the game Jacob!")
       
        if (done_model[model_number] == 1): # If we have already done this model (i.e., we kept the files and skipped this loop), move along.
            assert(FirstFile[model_number] == FirstFile[model_number - 1]) 
            assert(LastFile[model_number] == LastFile[model_number - 1]) 
            continue
        
        for fnr in range(FirstFile[model_number] + rank, LastFile[model_number]+1, size): # Divide up the input files across the processors.            

            GG, Gal_Desc = ReadScripts.ReadGals_SAGE(galaxies_filepath_array[model_number], fnr, number_snapshots[model_number], comm) # Read galaxies 
            G_Merged, _ = ReadScripts.ReadGals_SAGE(merged_galaxies_filepath_array[model_number], fnr, number_snapshots[model_number], comm) # Also need the merged galaxies.
            G = ReadScripts.Join_Arrays(GG, G_Merged, Gal_Desc) # Then join them together for all galaxies. 
            
            #pp = "post_quasar_SFR_STARBURST_QUASARWIND_{0}_snap_{1}".format(fnr, PlotSnapList[0][0])
            #plot_post_quasar_SFR(PlotSnapList, model_number, GG, pp) 
            #exit()
            ## These arrays are used to control the escape fraction that depends upon quasar activity. ##
            quasar_tracking = np.full(len(G), -1, dtype = quasar_tracking_dtype) # Initialize all the quasar tracking to -1.
            quasar_tracking['QuasarActivityToggle'][:] = 0.0 # Then set a few of the parameters to 0 rather than -1.
            quasar_tracking['QuasarBoostActiveTime'][:] = 0.0 
            quasar_tracking['QuasarFractionalPhoton'][:] = 0.0 
                            
            keep_files = 1 # Flips to 0 when we are done with this file.
            current_model_number = model_number # Used to differentiate between outer model_number and the inner model_number because we can keep files across model_numbers.

            while(keep_files == 1): 
                ## Just a few definitions to cut down the clutter a smidge. ##
                current_halo_cut = halo_cut[current_model_number]
                NumSubsteps = number_substeps[current_model_number]
                do_observed_LF = calculate_observed_LF[current_model_number]

                for snapshot_idx in range(0, len(SnapList[current_model_number])): # Now let's calculate stats for each required redshift.                
                    current_snap = SnapList[current_model_number][snapshot_idx] # Get rid of some clutter.

                    w_gal = np.where((G.GridHistory[:, current_snap] != -1) & (G.GridStellarMass[:, current_snap] > 0.0) & (G.LenHistory[:, current_snap] > current_halo_cut) & (G.GridSFR[:, current_snap] >= 0.0) & (G.GridFoFMass[:, current_snap] >= 0.0))[0] # Only include those galaxies that existed at the current snapshot, had positive (but not infinite) stellar/Halo mass and Star formation rate. Ensure the galaxies also resides in a halo that is sufficiently resolved.
                    w_merged_gal = np.where((G_Merged.GridHistory[:, current_snap] != -1) & (G_Merged.GridStellarMass[:, current_snap] > 0.0) & (G_Merged.LenHistory[:, current_snap] > current_halo_cut) & (G_Merged.GridSFR[:, current_snap] >= 0.0) & (G_Merged.GridFoFMass[:, current_snap] >= 0.0) & (G_Merged.LenMergerGal[:,current_snap] > current_halo_cut))[0] 
              
                    print("There were {0} galaxies for snapshot {1} (Redshift {2:.3f}) model {3}.".format(len(w_gal), current_snap, AllVars.SnapZ[current_snap], current_model_number))
                   
                    if (len(w_gal) == 0): 
                        continue

                    mass_gal = np.log10(G.GridStellarMass[w_gal, current_snap] * 1.0e10 / AllVars.Hubble_h) # Msun. Log Units.
                    SFR_gal = np.log10(G.GridSFR[w_gal, current_snap]) # Msun yr^-1.  Log Units.                                                       
                    halo_part_count = G.LenHistory[w_gal, current_snap]
                    metallicity_gal = G.GridZ[w_gal, current_snap]  
                    metallicity_tremonti_gal = np.log10(G.GridZ[w_gal, current_snap] / 0.02) + 9.0 # Using the Tremonti relationship for metallicity.
                    mass_central = np.log10(G.GridFoFMass[w_gal, current_snap] * 1.0e10 / AllVars.Hubble_h) # Msun. Log Units. 
                    ejected_fraction = G.EjectedFraction[w_gal, current_snap]

                    w_dust = np.where(((G.GridDustColdGas[w_gal, current_snap]
                                      +G.GridDustHotGas[w_gal, current_snap]
                                      +G.GridDustEjectedMass[w_gal, current_snap]) > 0.0) 
                                      & (G.GridType[w_gal, current_snap] == 0))[0]
        
                    total_dust_gal = np.log10((G.GridDustColdGas[w_gal[w_dust], current_snap] 
                                              +G.GridDustHotGas[w_gal[w_dust], current_snap]
                                              +G.GridDustEjectedMass[w_gal[w_dust], current_snap])
                                              * 1.0e10 / AllVars.Hubble_h)
                    mass_gal_dust = np.log10(G.GridStellarMass[w_gal[w_dust], current_snap] 
                                         * 1.0e10 / AllVars.Hubble_h)

                    mass_centralgal_dust = np.log10(G.GridFoFMass[w_gal[w_dust], current_snap] 
                                         * 1.0e10 / AllVars.Hubble_h)

                    fname="/lustre/projects/p004_swin/jseiler/kali/dust/npz_files/stellarmass_dust_snap{0:03d}_{1}" \
                            .format(current_snap, fnr)
                    #np.savez(fname, mass_gal_dust)


                    fname="/lustre/projects/p004_swin/jseiler/kali/dust/npz_files/halomass_dust_snap{0:03d}_{1}" \
                            .format(current_snap, fnr)
                    
                    #np.savez(fname, mass_centralgal_dust)

                    fname="/lustre/projects/p004_swin/jseiler/kali/dust/npz_files/dustmass_dust_snap{0:03d}_{1}" \
                            .format(current_snap, fnr)
                    #np.savez(fname, total_dust_gal)

                   
                    w_test = np.where(mass_centralgal_dust > 11.5)[0]
                    #print("Mass of Galaxy {0}".format(mass_gal_dust[w_test]))
                    #print("Mass of Halo {0}"\
                    #      .format(mass_centralgal_dust[w_test]))
                    #print("Mass of Dust {0}".format(total_dust_gal[w_test]))
                    #print("TreeNr {0}".format(G.TreeNr[w_gal[w_test]]))
                    #exit() 
                    reionmod = G.GridReionMod[w_gal, current_snap]
                    mass_reionmod_central = mass_central[reionmod > -1]
                    reionmod = reionmod[reionmod > -1] # Some satellite galaxies that don't have HotGas and hence won't be stripped. As a result reionmod = -1 for these. Ignore them.        

                    mass_BH = G.GridBHMass[w_gal, current_snap] * 1.0e10 / AllVars.Hubble_h # Msun. Not log units. 
                    
                    merge_flag = G_Merged.mergeType[w_merged_gal]
                    merge_flag[merge_flag >= 1] = 1.0 
                    merge_mass_central = np.log10(G_Merged.GridFoFMass[w_merged_gal, current_snap] * 1.0e10 / AllVars.Hubble_h) # Msun. Log Units. 
                    merge_mass_galaxy = np.log10(G_Merged.GridStellarMass[w_merged_gal, current_snap] * 1.0e10 / AllVars.Hubble_h) # Msun. Log Units. 
                                        
                    L_UV = SFR_gal + 39.927 # Using relationship from STARBURST99, units of erg s^-1 A^-1. Log Units.
                    M_UV = AllVars.Luminosity_to_ABMag(L_UV, 1600)

                    if (do_observed_LF == 1): # Calculate the UV extinction if requested. 
                        M_UV_obs = calculate_UV_extinction(AllVars.SnapZ[current_snap], L_UV, M_UV[snap_idx])
                
                    ## Here we do calculations involving the quasars temporarily boosting the escape fraction. ##            
                    if (fesc_prescription[current_model_number] == 3): # Only do it if this model is using this escape fraction prescription.
                        N_dyntime = fesc_normalization[current_model_number][2]
                        (quasar_tracking, N_quasars_tmp, N_quasars_boost_tmp, dynamicaltime_quasars_tmp) = do_quasar_tracking(quasar_tracking, N_dyntime, NumSubsteps, current_snap, w_gal, G.DynamicalTime, G.QuasarActivity, G.QuasarSubstep)       
                        dynamicaltime_all_tmp = []
                        dynamicaltime_all_tmp.extend(G.DynamicalTime[w_gal, current_snap])
                                                 
                        if (N_quasars_tmp > 0):           
                            (dynamicaltime_quasars_mean_z[current_model_number][snapshot_idx], dynamicaltime_quasars_std_z[current_model_number][snapshot_idx]) = update_cumulative_stats(dynamicaltime_quasars_mean_z[current_model_number][snapshot_idx], dynamicaltime_quasars_std_z[current_model_number][snapshot_idx], N_quasars_z[current_model_number][snapshot_idx], np.mean(dynamicaltime_quasars_tmp), np.std(dynamicaltime_quasars_tmp), N_quasars_tmp) # Update the trackin of the dynamical times. 

                        (dynamicaltime_all_mean_z[current_model_number][snapshot_idx], dynamicaltime_all_std_z[current_model_number][snapshot_idx]) = update_cumulative_stats(dynamicaltime_all_mean_z[current_model_number][snapshot_idx], dynamicaltime_all_std_z[current_model_number][snapshot_idx], N_z[current_model_number][snapshot_idx], np.mean(dynamicaltime_all_tmp), np.std(dynamicaltime_all_tmp), len(w_gal))
                                    
                        N_quasars_z[current_model_number][snapshot_idx] += N_quasars_tmp # Remember how many quasars went off.
                        N_quasars_boost_z[current_model_number][snapshot_idx] += N_quasars_boost_tmp # And how many galaxies are under the influence of the boost.
      
                    galaxy_halo_mass_mean_local, galaxy_halo_mass_std_local = Calculate_HaloPartStellarMass(halo_part_count, mass_gal, stellar_mass_halolen_lower[current_model_number], stellar_mass_halolen_upper[current_model_number]) # This is the average stellar mass for galaxies whose halos have the specified number of particles.
                    galaxy_halo_mass_mean[current_model_number][snapshot_idx] += pow(10, galaxy_halo_mass_mean_local) / (LastFile[current_model_number] + 1) # Adds to the average of the mean. 

                    fesc_local = calculate_fesc(fesc_prescription[current_model_number], fesc_normalization[current_model_number], mass_central, ejected_fraction, quasar_tracking['QuasarFractionalPhoton'][w_gal]) 

                         
                    photons_HI_gal = calculate_photons(SFR_gal, metallicity_gal) # photons s^-1. Log Units. 
                    photons_HI_gal_nonlog = [10**x for x in photons_HI_gal] # Turn to non-log.
                    ionizing_photons = np.multiply(photons_HI_gal_nonlog, fesc_local) # Then get the number that actually escape.

                    #for i in range(10):
                    #    print("SFR {0} Mass {1} Photons {2} fesc {3}".format(10**SFR_gal[i], mass_gal[i], photons_HI_gal[i], fesc_local[i]))                   

                    ###########################################                     
                    ######## BASE PROPERTIES CALCULATED #######
                    ###########################################

                    # Time to calculate relevant statistics.

                    ### Functions of Galaxies/Stellar Mass ###

                    ## Stellar Mass Function ##

                    (counts_local, bin_edges, bin_middle) = AllVars.Calculate_Histogram(mass_gal, bin_width, 0, m_gal_low, m_gal_high) # Bin the Stellar Mass 
                    SMF[current_model_number][snapshot_idx] += counts_local 

                    ## Escape Fraction ##                    

                    (mean_fesc_galaxy_local, std_fesc_galaxy_local, N_local, sum_fesc_galaxy, bin_middle) = AllVars.Calculate_2D_Mean(mass_gal, fesc_local, bin_width, m_gal_low, m_gal_high)
                    (mean_fesc_galaxy_array[current_model_number][snapshot_idx], std_fesc_galaxy_array[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_fesc_galaxy_array[current_model_number][snapshot_idx], std_fesc_galaxy_array[current_model_number][snapshot_idx], N_galaxy_array[current_model_number][snapshot_idx], mean_fesc_galaxy_local, std_fesc_galaxy_local, N_local) 
 
                    ## Black Hole Mass ##
 
                    (mean_BHmass_galaxy_local, std_BHmass_galaxy_local, N_local, sum_BHmass_galaxy, bin_middle) = AllVars.Calculate_2D_Mean(mass_gal, mass_BH, bin_width, m_gal_low, m_gal_high) 
                    (mean_BHmass_galaxy_array[current_model_number][snapshot_idx], std_BHmass_galaxy_array[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_BHmass_galaxy_array[current_model_number][snapshot_idx], std_BHmass_galaxy_array[current_model_number][snapshot_idx], N_galaxy_array[current_model_number][snapshot_idx], mean_BHmass_galaxy_local, std_BHmass_galaxy_local, N_local) 
                     
                    ## Total Dust Mass ##
                    (mean_dust_galaxy_local, std_dust_galaxy_local, N_local,
                     sum_dust_galaxy, bin_middle) = AllVars.Calculate_2D_Mean(
                                                    mass_gal_dust, total_dust_gal,
                                                    bin_width, m_gal_low,
                                                    m_gal_high) 

                    (mean_dust_galaxy_array[current_model_number][snapshot_idx],
                     std_dust_galaxy_array[current_model_number][snapshot_idx]) = \
                    update_cumulative_stats(mean_dust_galaxy_array[current_model_number][snapshot_idx],
                                            std_dust_galaxy_array[current_model_number][snapshot_idx],
                                            N_galaxy_array[current_model_number][snapshot_idx],
                                            mean_dust_galaxy_local,
                                            std_dust_galaxy_local,
                                            N_local) 

                    N_galaxy_array[current_model_number][snapshot_idx] += N_local 

                    ### Functions of Halos/Halo Mass ###

                    ## Ejected Fraction ##

                    (mean_ejected_halo_local, std_ejected_halo_local, N_local, sum_ejected_halo, bin_middle) = AllVars.Calculate_2D_Mean(mass_central, ejected_fraction, bin_width, m_low, m_high) 
                    (mean_ejected_halo_array[current_model_number][snapshot_idx], std_ejected_halo_array[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_ejected_halo_array[current_model_number][snapshot_idx], std_ejected_halo_array[current_model_number][snapshot_idx], N_halo_array[current_model_number][snapshot_idx], mean_ejected_halo_local, std_ejected_halo_local, N_local) # Then update the running total.

                    ## Quasar Fraction ##

                    (mean_quasar_activity_local, std_quasar_activity_local, N_local, sum_ejected_halo, bin_middle) = AllVars.Calculate_2D_Mean(mass_central, G.QuasarActivity[w_gal, current_snap], bin_width, m_low, m_high) 
                    (mean_quasar_activity_array[current_model_number][snapshot_idx], std_quasar_activity_array[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_quasar_activity_array[current_model_number][snapshot_idx], std_quasar_activity_array[current_model_number][snapshot_idx], N_halo_array[current_model_number][snapshot_idx], mean_quasar_activity_local, std_quasar_activity_local, N_local) # Then update the running total.
                   
                    ## fesc Value ##

                    (mean_fesc_halo_local, std_fesc_halo_local, N_local, sum_ejected_halo, bin_middle) = AllVars.Calculate_2D_Mean(mass_central, fesc_local, bin_width, m_low, m_high) 
                    (mean_fesc_halo_array[current_model_number][snapshot_idx], std_fesc_halo_array[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_fesc_halo_array[current_model_number][snapshot_idx], std_fesc_halo_array[current_model_number][snapshot_idx], N_halo_array[current_model_number][snapshot_idx], mean_fesc_halo_local, std_fesc_halo_local, N_local) # Then update the running total. 

                    ## Ngamma ##

                    (mean_Ngamma_halo_local, std_Ngamma_halo_local, N_local, sum_ejected_halo, bin_middle) = AllVars.Calculate_2D_Mean(mass_central, ionizing_photons, bin_width, m_low, m_high)  

                    mean_Ngamma_halo_local = np.divide(mean_Ngamma_halo_local, 1.0e50) ## Divide out a constant to keep the numbers manageable.
                    std_Ngamma_halo_local = np.divide(std_Ngamma_halo_local, 1.0e50)

                    (mean_Ngamma_halo_array[current_model_number][snapshot_idx], std_Ngamma_halo_array[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_Ngamma_halo_array[current_model_number][snapshot_idx], std_Ngamma_halo_array[current_model_number][snapshot_idx], N_halo_array[current_model_number][snapshot_idx], mean_Ngamma_halo_local, std_Ngamma_halo_local, N_local) # Then update the running total. 

                    ## Reionization Modifier ##

                    (mean_reionmod_halo_local, std_reionmod_halo_local, N_local, sum_reionmod_halo, bin_middle) = AllVars.Calculate_2D_Mean(mass_reionmod_central, reionmod, bin_width, m_low, m_high) 
                    (mean_reionmod_halo_array[current_model_number][snapshot_idx], std_reionmod_halo_array[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_reionmod_halo_array[current_model_number][snapshot_idx], std_reionmod_halo_array[current_model_number][snapshot_idx], N_halo_array[current_model_number][snapshot_idx], mean_reionmod_halo_local, std_reionmod_halo_local, N_local) # Then update the running total. 

                    ## Mergers ##

                    if (len(merge_flag) > 0):
                        (_, _, _, merger_counts_local, _) = AllVars.Calculate_2D_Mean(merge_mass_central, merge_flag, bin_width, m_low, m_high)
                        mergers_halo_array[model_number][snapshot_idx] += merger_counts_local

                        (_, _, _, merger_counts_local_galaxy, _) = AllVars.Calculate_2D_Mean(merge_mass_galaxy, merge_flag, bin_width, m_gal_low, m_gal_high)
                        mergers_galaxy_array[model_number][snapshot_idx] += merger_counts_local_galaxy
        
                    ## Total Dust Mass ##

                    (mean_dust_halo_local, std_dust_halo_local, N_local,
                     sum_dust_halo, bin_middle) = AllVars.Calculate_2D_Mean(
                                                    mass_centralgal_dust, total_dust_gal,
                                                    bin_width, m_low,
                                                    m_high) 

                    #print("Halo {0}".format(mean_dust_halo_local))
                    #print("Galaxy {0}".format(mean_dust_galaxy_local))

                    (mean_dust_halo_array[current_model_number][snapshot_idx],
                     std_dust_halo_array[current_model_number][snapshot_idx]) = \
                    update_cumulative_stats(mean_dust_halo_array[current_model_number][snapshot_idx],
                                            std_dust_halo_array[current_model_number][snapshot_idx],
                                            N_halo_array[current_model_number][snapshot_idx],
                                            mean_dust_halo_local,
                                            std_dust_halo_local,
                                            N_local) 

                                
                    N_halo_array[current_model_number][snapshot_idx] += N_local                         

                    ### Functions of redshift ###

                    ## Ngamma ##

                    sum_Ngamma_z_array[current_model_number][snapshot_idx] += np.sum(np.divide(ionizing_photons, 1.0e50)) # Remember that we're dividing out a constant! 

                    ## fesc Value ## 
                    (mean_fesc_z_array[current_model_number][snapshot_idx], std_fesc_z_array[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_fesc_z_array[current_model_number][snapshot_idx], std_fesc_z_array[current_model_number][snapshot_idx], N_z[current_model_number][snapshot_idx], np.mean(fesc_local), np.std(fesc_local), len(w_gal)) # Updates the mean escape fraction for this redshift.

                    ## Reionization Modifier ##
                    (mean_reionmod_z[current_model_number][snapshot_idx], std_reionmod_z[current_model_number][snapshot_idx]) = update_cumulative_stats(mean_reionmod_z[current_model_number][snapshot_idx], std_reionmod_z[current_model_number][snapshot_idx], N_reionmod_z[current_model_number][snapshot_idx], np.mean(reionmod), np.std(reionmod), len(reionmod))
                    N_reionmod_z[current_model_number][snapshot_idx] += len(reionmod)
         
                    N_z[current_model_number][snapshot_idx] += len(w_gal)
                                
                done_model[current_model_number] = 1
                if (current_model_number < number_models):                
                    keep_files =  same_files[current_model_number] # Decide if we want to keep the files loaded or throw them out. 
                    current_model_number += 1 # Update the inner loop model number.
   
    StellarMassFunction(PlotSnapList, SMF, simulation_norm, FirstFile,
                        LastFile, NumFile, galaxy_halo_mass_mean, model_tags, 
                        1, paper_plots, "BHmodels")
    #plot_reionmod(PlotSnapList, SnapList, simulation_norm, mean_reionmod_halo_array, 
                  #std_reionmod_halo_array, N_halo_array, mean_reionmod_z, 
                  #std_reionmod_z, N_reionmod_z, False, model_tags,
                  #"reionmod_selfcon")
    #plot_dust_scatter(SnapList, mass_gal_dust, mass_centralgal_dust, total_dust_gal, 
    #                  "dust_scatter") 
    #plot_dust(PlotSnapList, SnapList, simulation_norm, mean_dust_galaxy_array,
    #          std_dust_galaxy_array, N_galaxy_array, mean_dust_halo_array,
    #          std_dust_halo_array, N_halo_array, False, model_tags,
    #          "dustmass_total")
    #plot_stellarmass_blackhole(PlotSnapList, simulation_norm, mean_BHmass_galaxy_array, std_BHmass_galaxy_array, N_galaxy_array, model_tags, "StellarMass_BHMass")
    #plot_ejectedfraction(SnapList, mean_ejected_halo_array, std_ejected_halo_array, N_halo_array, model_tags, "tiamat_newDelayedComp_ejectedfract_highz") ## PARALELL COMPATIBLE # Ejected fraction as a function of Halo Mass 
    #plot_fesc(SnapList, mean_fesc_z_array, std_fesc_z_array, N_z, model_tags, "Quasarfesc_z_DynamicalTimes") ## PARALELL COMPATIBLE 
    #plot_quasars_count(SnapList, PlotSnapList, N_quasars_z, N_quasars_boost_z, N_z, mean_quasar_activity_array, std_quasar_activity_array, N_halo_array, mergers_halo_array, SMF, mergers_galaxy_array, fesc_prescription, simulation_norm, FirstFile, LastFile, NumFile, model_tags, "SN_Prescription")
    #plot_fesc_galaxy(SnapList, PlotSnapList, simulation_norm, mean_fesc_galaxy_array, std_fesc_galaxy_array, N_galaxy_array, mean_fesc_halo_array, std_fesc_halo_array,  N_halo_array, galaxy_halo_mass_mean, model_tags, "fesc_test") 
    #plot_photoncount(SnapList, sum_Ngamma_z_array, simulation_norm, FirstFile, LastFile, NumFile, model_tags, "Ngamma_test") ## PARALELL COMPATIBLE
    #plot_mvir_Ngamma(SnapList, mean_Ngamma_halo_array, std_Ngamma_halo_array, N_halo_array, model_tags, "Mvir_Ngamma_test", fesc_prescription, fesc_normalization, "/lustre/projects/p004_swin/jseiler/tiamat/halo_ngamma/") ## PARALELL COMPATIBLE 

