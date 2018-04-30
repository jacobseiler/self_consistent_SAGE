#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import sys
import os
import numpy as np
import pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from numpy import *
from random import sample, seed
import math
import random
import csv
from io import StringIO
from collections import Counter
from matplotlib.colors import LogNorm
import time
from matplotlib.ticker import MultipleLocator
from os.path import getsize as getFileSize
from numpy import inf 
import h5py

np.set_printoptions(threshold=np.nan)
def Read_SAGE_Objects(Model_Name, Object_Desc, Contain_TreeInfo, Dot, fnr, comm=None):
    # Initialize variables.
    TotNTrees = 0
    TotNHalos = 0
    FileIndexRanges = []
   
    if comm is not None: 
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1

    print("Determining array storage requirements.")
    
    # Read each file and determine the total number of galaxies to be read in
    goodfiles = 0

    if (Dot == 1):
        fname = Model_Name+'.'+str(fnr)  # Complete filename
    else:
        fname = Model_Name+'_'+str(fnr)  # Complete filename

    if not os.path.isfile(fname):
        print("File\t%s  \tdoes not exist!  Skipping..." % (fname))
        quit() 
        
    if getFileSize(fname) == 0:
        print("File\t%s  \tis empty!  Skipping..." % (fname))
        quit() 
        
    fin = open(fname, 'rb')  # Open the file
    Nsubsteps = np.fromfile(fin, np.dtype(np.int32),1) 
    Nsnap = np.fromfile(fin, np.dtype(np.int32),1) 
    redshifts = np.fromfile(fin, np.dtype(np.float64), int(Nsnap)) 
    Hubble_h = np.fromfile(fin, np.dtype(np.float64),1) 
    Omega = np.fromfile(fin, np.dtype(np.float64),1) 
    OmegaLambda = np.fromfile(fin, np.dtype(np.float64),1) 
    BaryonFrac = np.fromfile(fin, np.dtype(np.float64),1) 
    PartMass = np.fromfile(fin, np.dtype(np.float64),1) 
    BoxSize = np.fromfile(fin, np.dtype(np.float64),1) 
    GridSize = np.fromfile(fin, np.dtype(np.int32),1) 

    if (Contain_TreeInfo == 1):
        Ntrees = np.fromfile(fin,np.dtype(np.int32),1)  # Read number of trees in file
        TotNTrees = TotNTrees + Ntrees  # Update total sim trees number
    NtotHalos = np.fromfile(fin,np.dtype(np.int32),1)[0]  # Read number of gals in file.
    TotNHalos = TotNHalos + NtotHalos  # Update total sim gals number
    print("Rank %d is reading File %s and contains %d total galaxies" %(rank, fname, NtotHalos))
	
    goodfiles = goodfiles + 1  # Update number of files read for volume calculation
    fin.close()
    
    print("")
    print("Input files contain:\t%d trees ;\t%d Halos/Galaxies." % (TotNTrees, TotNHalos))
    print("")

    # Initialize the storage array
    G = np.empty(TotNHalos, dtype=Object_Desc)
    
    offset = 0  # Offset index for storage array
    
    # Open each file in turn and read in the preamble variables and structure.
    print("Reading in files.")
    if (Dot == 1):
        fname = Model_Name+'.'+str(fnr)  # Complete filename
    else:
        fname = Model_Name+'_'+str(fnr)  # Complete filename
        
    if not os.path.isfile(fname):
        print("Couldn't find file %s" %(fname))
        exit()
        
    if getFileSize(fname) == 0:
        print("File %s is empty!" %(fname))
        exit()
    
    fin = open(fname, 'rb')  # Open the file
    Nsubsteps = np.fromfile(fin, np.dtype(np.int32),1) 
    Nsnap = np.fromfile(fin, np.dtype(np.int32),1) 
    redshifts = np.fromfile(fin, np.dtype(np.float64), int(Nsnap))
    Hubble_h = np.fromfile(fin, np.dtype(np.float64),1) 
    Omega = np.fromfile(fin, np.dtype(np.float64),1) 
    OmegaLambda = np.fromfile(fin, np.dtype(np.float64),1) 
    BaryonFrac = np.fromfile(fin, np.dtype(np.float64),1) 
    PartMass = np.fromfile(fin, np.dtype(np.float64),1) 
    BoxSize = np.fromfile(fin, np.dtype(np.float64),1) 
    GridSize = np.fromfile(fin, np.dtype(np.int32),1) 


    if (Contain_TreeInfo == 1):
        Ntrees = np.fromfile(fin, np.dtype(np.int32), 1)  # Read number of trees in file
    NtotHalos = np.fromfile(fin, np.dtype(np.int32), 1)[0]  # Read number of gals in file.
    if (Contain_TreeInfo == 1):
        GalsPerTree = np.fromfile(fin, np.dtype((np.int32, Ntrees)),1) # Read the number of gals in each tree

    print(":Rank %d is Reading N= %d Objects from %s: " %(rank, NtotHalos, fname))
 	
    GG = np.fromfile(fin, Object_Desc, NtotHalos)  # Read in the galaxy structures

    FileIndexRanges.append((offset,offset+NtotHalos))
      
    G = GG.view(np.recarray)
    return G

def ReadHalos(DirName, First_File, Last_File):

    Halo_Desc_full = [
    ('Descendant',          np.int32),
    ('FirstProgenitor',     np.int32),
    ('NextProgenitor',      np.int32),
    ('FirstHaloInFOFgroup', np.int32),
    ('NextHaloInFOFgroup',  np.int32),
    ('Len',                 np.int32),
    ('M_mean200',           np.float32),
    ('Mvir',                np.float32),
    ('M_TopHat',            np.float32),
    ('Pos',                 (np.float32, 3)),
    ('Vel',                 (np.float32, 3)),
    ('VelDisp',             np.float32),
    ('Vmax',                np.float32),
    ('Spin',                (np.float32, 3)),
    ('MostBoundID',         np.int64),
    ('SnapNum',             np.int32),
    ('Filenr',              np.int32),
    ('SubHaloIndex',        np.int32),
    ('SubHalfMass',         np.float32)
                     ]

    print("First_File = %d" %(First_File))
    print("Last_File = %d" %(Last_File))

    names = [Halo_Desc_full[i][0] for i in range(len(Halo_Desc_full))]
    formats = [Halo_Desc_full[i][1] for i in range(len(Halo_Desc_full))]
    Halo_Desc = np.dtype({'names':names, 'formats':formats}, align=True)

    return Read_SAGE_Objects(DirName, Halo_Desc, 1, 1, First_File, Last_File)
    
## Using this one ##   
def ReadGals_SAGE(DirName, fnr, MAXSNAPS, comm=None):

    Galdesc_full = [ 
         ('TreeNr', np.int32),
         ('GridType', (np.int32, MAXSNAPS)),
         ('GridFoFHaloNr', (np.int32, MAXSNAPS)),
         ('GridHistory', (np.int32, MAXSNAPS)), 
         ('GridColdGas', (np.float32, MAXSNAPS)),
         ('GridHotGas', (np.float32, MAXSNAPS)),
         ('GridEjectedMass', (np.float32, MAXSNAPS)),
         ('GridDustColdGas', (np.float32, MAXSNAPS)),
         ('GridDustHotGas', (np.float32, MAXSNAPS)),
         ('GridDustEjectedMass', (np.float32, MAXSNAPS)),
         ('GridBHMass', (np.float32, MAXSNAPS)),
         ('GridStellarMass', (np.float32, MAXSNAPS)),
         ('GridSFR', (np.float32, MAXSNAPS)),
         ('GridZ', (np.float32, MAXSNAPS)),
         ('GridFoFMass', (np.float32, MAXSNAPS)),
         ('EjectedFraction', (np.float32, MAXSNAPS)),  
         ('LenHistory', (np.int32, MAXSNAPS)),
         ('QuasarActivity', (np.int32, MAXSNAPS)),
         ('QuasarSubstep', (np.int32, MAXSNAPS)),
         ('DynamicalTime', (np.float32, MAXSNAPS)),
         ('LenMergerGal', (np.int32, MAXSNAPS)),
         ('GridReionMod', (np.float32, MAXSNAPS)),
         ('GridNgamma_HI', (np.float32, MAXSNAPS)),
         ('Gridfesc', (np.float32, MAXSNAPS))
         ]
   
    print("Reading in SAGE files (Post STARBURST).")

    names = [Galdesc_full[i][0] for i in range(len(Galdesc_full))]
    formats = [Galdesc_full[i][1] for i in range(len(Galdesc_full))] 
    Gal_Desc = np.dtype({'names':names, 'formats':formats}, align=True)  
 
    return (Read_SAGE_Objects(DirName, Gal_Desc, 1, 0, fnr, comm), Gal_Desc)

def Join_Arrays(Array1, Array2, Desc):
    
   
    G = np.empty(len(Array1) + len(Array2), Desc) # Create an empty array with enough space to hold both arrays.

    G[0:len(Array1)] = Array1[0:len(Array1)].copy() # Slice in the first array.
    G[len(Array1):len(Array1) + len(Array2)] = Array2[0:len(Array2)].copy() # Then append in the second array.

    G = G.view(np.recarray) # Turn into a C-like struct.

    return G

def Create_Snap_Arrays(G, NumSnaps, SnapList):

    print("Creating separate arrays for Snapshots", SnapList)

    SnapCount = np.zeros((NumSnaps))

    for i in range(0, len(G)):
        for j in range(0,NumSnaps):
            if (G.GridHistory[i,j] != -1):
                SnapCount[j] += 1


def read_binary_grid(filepath, GridSize, precision, reshape=True):
    '''
    Reads a cubic, Cartesian grid that was stored in binary.
    NOTE: Assumes the grid has equal number of cells in each dimension.

    Parameters
    ----------
    filepath : string
        Location of the grid file
    GridSize : integer
        Number of cells along one dimension.  Grid is assumed to be saved in the form N*N*N. 
    precision : integer
        Denotes the precision of the data being read in.
        0 : Integer (4 bytes)
        1 : Float (4 bytes)
        2 : Double (8 bytes)
    reshape : boolean
        Controls whether the array should be reshaped into a cubic array of shape (GridSize, GridSize, GridSize) or kepts as a 1D array.
        Default: True.

    Returns
    -------
    grid : `np.darray'
	The read in grid as a numpy object.  Shape will be N*N*N.
    '''
    print("Reading binary grid %s with precision option %d" %(filepath, precision)) 

    ## Set the format the input file is in. ##
    readformat = 'None'
    if precision == 0:
        readformat = np.int32
        byte_size = 4
    elif precision == 1:
        readformat = np.float32
        byte_size = 4
    elif precision == 2: 
        readformat = np.float64
        byte_size = 8
    else:
        print("You specified a read format of %d" %(precision))
        raise ValueError("Only 0, 1, 2 (corresponding to integers, float or doubles respectively) are currently supported.")

    ## Check that the file is the correct size. ##
    filesize = os.stat(filepath).st_size
    if(GridSize*GridSize*GridSize * byte_size != filesize):
        print("The size of the file is %d bytes whereas we expected it to be %d bytes" %(filesize, GridSize*GridSize*GridSize * byte_size))
        raise ValueError("Mismatch between size of file and expected size.")

    fd = open(filepath, 'rb')
    grid = np.fromfile(fd, count = GridSize**3, dtype = readformat) 
    if (reshape == True):
        grid.shape = (GridSize, GridSize, GridSize) 
    fd.close()

    return grid

def read_meraxes_hdf5():
    hdf5_file_name = '/home/msinha/scratch/tao/data_products/output/meraxes/tiamat/meraxes.hdf5'
    dataset = '/home/msinha/scratch/tao/data_products/output/meraxes/tiamat/meraxes_0.hdf5'

    container = h5py.File(hdf5_file_name, 'r')
    print("Keys: %s" %(container.keys()))
    
def read_trees_smallarray(treedir, file_idx, simulation):
    """
    Reads a single file of halos into an array.
    Assumes the tree are named as '<tree_dir>/subgroup_trees_<file_idx>.dat' where file_idx is padded out to 3 digits or '<tree_dir>/lhalotree.bin.<file_idx>' depending on the simulation. 

    Parameters
    ==========

    treedir : string
        Base directory path for the trees.
    file_idx : int
        File number we are reeding in.   
    simulation : int
        Simulation we are reading for.  Determines the naming convention for the files.
        0 : Pip (Britton's Simulation) built using Greg's code
        1 : Tiamat
        2 : Manodeep's 1024 Simulation
        3 : Pip built using Rockstar.
        4 : Kali built using Greg's code.

    Returns
    =======

    Halos : array of halos with data-type specified by 'Halo_Desc_full'
        The read in halos for this file. 
    HalosPerTree : array of ints
        Number of halos within each tree of the file.     
    """ 

    Halo_Desc_full = [
    ('Descendant',          np.int32),
    ('FirstProgenitor',     np.int32),
    ('NextProgenitor',      np.int32),
    ('FirstHaloInFOFgroup', np.int32),
    ('NextHaloInFOFgroup',  np.int32),
    ('Len',                 np.int32),
    ('M_mean200',           np.float32),
    ('Mvir',                np.float32),
    ('M_TopHat',            np.float32),
    ('Pos',                 (np.float32, 3)),
    ('Vel',                 (np.float32, 3)),
    ('VelDisp',             np.float32),
    ('Vmax',                np.float32),
    ('Spin',                (np.float32, 3)),
    ('MostBoundID',         np.int64),
    ('SnapNum',             np.int32),
    ('Filenr',              np.int32),
    ('SubHaloIndex',        np.int32),
    ('SubHalfMass',         np.float32)
                     ]

    names = [Halo_Desc_full[i][0] for i in range(len(Halo_Desc_full))]
    formats = [Halo_Desc_full[i][1] for i in range(len(Halo_Desc_full))]
    Halo_Desc = np.dtype({'names':names, 'formats':formats}, align=True)

    if (simulation == 0 or simulation == 1 or simulation == 4):
        fname = "{0}/subgroup_trees_{1:03d}.dat".format(treedir, file_idx)
    elif (simulation == 2 or simulation == 3):
        fname = "{0}/lhalotree.bin.{1}".format(treedir, file_idx)
    else:
        raise ValueError("Invalid simulation option chosen.")

    print("Reading for file {0}".format(fname)) 
    fin = open(fname, 'rb')  # Open the file

    trees_thisfile = np.fromfile(fin,np.dtype(np.int32),1)  # Read number of trees in file.
    halos_thisfile = np.fromfile(fin,np.dtype(np.int32),1)[0]  # Read number of halos in file.

    HalosPerTree = np.fromfile(fin, np.dtype((np.int32, trees_thisfile)),1)[0] # Read the number of halos in each tree.

    Halos = np.fromfile(fin, Halo_Desc, halos_thisfile)  # Read in the halos.

    fin.close() # Close the file  

    return Halos, HalosPerTree

def load_data(fname):
    """
    Reads data from a .npz file.
    If no .npz file exists the function searches for a .txt file.  If the .txt file exists it saves it as a .npz file before return the requested data.
    If no .txt file exists return a FileNotFoundError.
 
    Parameters
    ==========

    fname : string
        Base name of the file (no extensions) 
        
    Returns
    =======

    data : array-like
        Array of the read data. Shape will be dependant upon how the file itself was saved.  
    """ 

    try:
        filename = "{0}.npz".format(fname)
        data = np.load(filename)

    except FileNotFoundError:

        print(".npz file does not exist, checking for a .txt file")
        filename = "{0}.txt".format(fname)
        try:
            data = np.loadtxt(filename)

        except FileNotFoundError:

            raise FileNotFoundError("File {0} could not be found".format(filename))           
        else:

            print(".txt file was successfully located and loaded.")
            print("Now saving as a .npz file")
                
            np.savez(fname, data)

            return load_data(fname) 
            
    else:               
        return data['arr_0']

def read_SAGE_ini(fname):
    """
    
    Reads the SAGE .ini file into a structured array.

    Parameters
    ----------

    fname: String. Required. 
        Path to the .ini file.         

    Returns
    ---------

    SAGE_desc: Numpy structured array.
        Structured array containing the SAGE .ini parameters. 

    names: Array-like of Strings.
        Names that index the Numpy structured array. 
         
    """

    SAGE_params_full = [ 
         ('FileNameGalaxies', '<U1024'),
         ('OutputDir', '<U1024'),
         ('GridOutputDir', '<U1024'),
         ('FirstFile', np.int32),
         ('LastFile', np.int32),
         ('NumOutputs', np.int32),
         ('LowSnap', np.int32),
         ('HighSnap', np.int32),
         ('TreeName', '<U1024'),
         ('TreeExtension', '<U1024'),
         ('SimulationDir', '<U1024'),
         ('FileWithSnapList', '<U1024'),
         ('LastSnapShotNr', '<U1024'),
         ('Omega', np.float64),
         ('OmegaLambda', np.float64),
         ('BaryonFrac', np.float64),
         ('Hubble_h', np.float64),
         ('PartMass', np.float64),
         ('BoxSize', np.float64),
         ('GridSize', np.int64),
         ('self_consistent', np.int64),
         ('SFprescription', np.int64),
         ('AGNrecipeOn', np.int64),
         ('SupernovaRecipeOn', np.int64),
         ('ReionizationOn', np.int64),
         ('DiskInstabilityOn', np.int64),
         ('SfrEfficiency', np.float64),
         ('FeedbackReheatingEpsilon', np.float64),
         ('FeedbackEjectionEfficiency', np.float64),
         ('IRA', np.int64),
         ('TimeResolutionSN', np.float64),
         ('ReIncorporationFactor', np.float64),
         ('RadioModeEfficiency', np.float64),
         ('QuasarModeEfficiency', np.float64),
         ('BlackHoleGrowthRate', np.float64),
         ('ThreshMajorMerger', np.float64),
         ('ThresholdSatDisruption', np.float64),
         ('Yield', np.float64),
         ('RecycleFraction', np.float64),
         ('FracZleaveDisk', np.float64),
         ('Reionization_z0', np.float64),
         ('Reionization_zr', np.float64),
         ('EnergySN', np.float64),
         ('RescaleSN', np.float64),
         ('IMF', np.int32),         
         ('PhotoionDir', '<U1024'),
         ('PhotoionName', '<U1024'),
         ('ReionRedshiftName', '<U1024'),
         ('ReionSnap', np.int32),
         ('PhotonPrescription', np.int32),
         ('Verbose', np.int32),
         ('fescPrescription', np.int32),            
         ('MH_low', np.float64),
         ('fesc_low', np.float64),
         ('MH_high', np.float64),
         ('fesc_high', np.float64),
         ('alpha', np.float64),
         ('beta', np.float64),
         ('fesc', np.float64),
         ('quasar_baseline', np.float64),
         ('quasar_boosted', np.float64),
         ('N_dyntime', np.float64),
         ('HaloPartCut', np.int32),
         ('UnitLength_in_cm', np.float64),
         ('UnitMass_in_g', np.float64),
         ('UnitVelocity_in_cm_per_s', np.float64)
         ]
    
                            
    print("Reading in SAGE ini file") 

    names = [SAGE_params_full[i][0] for i in range(len(SAGE_params_full))]
    formats = [SAGE_params_full[i][1] for i in range(len(SAGE_params_full))]
    SAGE_desc = np.empty(1, dtype = {'names':names, 'formats':formats}) 
    
    try:
        with open (fname, "r") as SAGE_file:
            data = SAGE_file.readlines()
            count = 0
            for line in range(len(data)):

                if (data[line][0] == "%" or data[line][0] == "\n"):                
                    continue
                try:
                    SAGE_desc[names[count]] = (data[line].split())[1]
                except ValueError: 
                    print("Current SAGE_desc is {0}".format(SAGE_desc))
                count += 1
        return SAGE_desc, names

    except FileNotFoundError:
        print("Could not file SAGE ini file {0}".format(fname))

def read_cifog_ini(fname):
    """
    
    Reads the cifog .ini file into a structured array.
    Since the cifog .ini file has section headers (e.g., '[General]') that are required for the
    cifog C code, we also generate a dictionary so we can reconstruct the .ini file with the 
    headers present. 

    Parameters
    ----------

    fname: String. Required. 
        Path to the .ini file.         

    Returns
    ---------

    cifog_desc: Numpy structured array.
        Structured array containing the cifog .ini parameters. 

    names: Array-like of Strings.
        Names that index the Numpy structured array. 

    headers: Dictionary.
        Dictionary of section headers keyed by the parameter name following the section header.
        E.g., the section '[General]' is before 'calcIonHistory' so headers['calcIonHistory'] =
        '[General]'.
         
    """

    cifog_params_full = [ 
         ('calcIonHistory', np.int32),
         ('numSnapshots', np.int32),
         ('stopSnapshot', np.int32),
         ('redshiftFile', '<U1024'),
         ('redshift_prevSnapshot', np.float64),
         ('finalRedshift', np.float64),
         ('evolutionTime', np.float64),
         ('size_linear_scale', np.float64),
         ('first_increment_in_logscale', np.float64),
         ('max_scale', np.float64),
         ('useDefaultMeanDensity', np.int32),
         ('useIonizeSphereModel', np.int32),
         ('useWebModel', np.int32), 
         ('photHImodel', np.int32),
         ('calcMeanFreePath', np.int32),
         ('constantRecombinations', np.int32),
         ('calcRecombinations', np.int32),
         ('solveForHelium', np.int32),
         ('paddedBox', np.float64),
         ('gridsize', np.int32),
         ('boxsize', np.float64),
         ('densityFilesAreInDoublePrecision', np.int32),
         ('nionFilesAreInDoublePrecision', np.int32),
         ('inputFilesAreComoving', np.int32),
         ('inputFilesAreSimulation', np.int32),
         ('SimulationLowSnap', np.int32),
         ('SimulationHighSnap', np.int32),
         ('inputIgmDensityFile', '<U1024'),
         ('densityInOverdensity', np.int32),
         ('meanDensity', np.float64),
         ('inputIgmClumpFile', '<U1024'),
         ('inputSourcesFile', '<U1024'),
         ('inputNionFile', '<U1024'),
         ('nion_factor', np.float64),
         ('output_XHII_file', '<U1024'),
         ('write_photHI_file', np.int32),
         ('output_photHI_file', '<U1024'),
         ('output_restart_file', '<U1024'),
         ('hubble_h', np.float64),
         ('omega_b', np.float64),
         ('omega_m', np.float64),
         ('omega_l', np.float64),
         ('sigma8', np.float64),
         ('Y', np.float64),         
         ('photHI_bg_file', '<U1024'),
         ('photHI_bg', np.float64),
         ('meanFreePathInIonizedMedium', np.float64),
         ('sourceSlopeIndex', np.float64),
         ('dnrec_dt', np.float64),
         ('recombinationTable', '<U1024'),
         ('zmin', np.float64),            
         ('zmax', np.float64),
         ('dz', np.float64),
         ('fmin', np.float64),
         ('fmax', np.float64),
         ('df', np.float64),
         ('dcellmin', np.float64),
         ('dcellmax', np.float64),
         ('ddcell', np.float64),
         ('inputSourcesHeIFile', '<U1024'),
         ('inputNionHeIFile', '<U1024'),
         ('inputSourcesHeIIFile', '<U1024'),
         ('inputNionHeIIFile', '<U1024'),
         ('dnrec_HeI_dt', np.float64),
         ('dnrec_HeII_dt', np.float64),
         ('output_XHeII_file', '<U1024'),
         ('output_XHeIII_file', '<U1024')
         ]
    
                            
    print("Reading in cifog ini file") 

    names = [cifog_params_full[i][0] for i in range(len(cifog_params_full))]
    formats = [cifog_params_full[i][1] for i in range(len(cifog_params_full))]
    cifog_desc = np.empty(1, dtype = {'names':names, 'formats':formats}) 
    
    headers = {}

    try:
        with open (fname, "r") as cifog_file:
            data = cifog_file.readlines()
            count = 0
            for line in range(len(data)):

                if (data[line][0] == "%" or data[line][0] == "\n"):                
                    continue

                if (data[line][0] == "["):
                    headers[names[count]] = data[line]
                    continue 

                cifog_desc[names[count]] = (data[line].split())[2]
                count += 1



        return cifog_desc, names, headers

    except FileNotFoundError:
        print("Could not file SAGE ini file {0}".format(fname))
 
