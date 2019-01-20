#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import numpy as np
import math


def Read_SAGE_header(model_name, fnr):

    if fnr:
        if (Dot == 1):
            fname = "{0}.{1}".format(model_name, fnr)
        else:
            fname = "{0}_{1}".format(model_name, fnr)
    else:
        fname = "{0}".format(model_name)
    if not os.path.isfile(fname):
        print("File {0} does not exist!".format(fname))
        raise RuntimeError

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
    
    Ntrees = np.fromfile(fin,np.dtype(np.int32),1)[0]  # Read number of trees in file

    fin.close()

    return Ntrees

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

    if fnr is not None:
        if (Dot == 1):
            fname = "{0}.{1}".format(Model_Name, fnr)
        else:
            fname = "{0}_{1}".format(Model_Name, fnr)
    else:
        fname = "{0}".format(Model_Name)
    if not os.path.isfile(fname):
        print("File\t%s  \tdoes not exist!  Skipping..." % (fname))
        raise RuntimeError

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
    GalsPerTree = np.fromfile(fin, np.dtype((np.int32, Ntrees)),1) 

    GG = np.fromfile(fin, Object_Desc, NtotHalos)  # Read in the galaxy structures      
    G = GG.view(np.recarray)

    return G

def read_trees(fname):

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


    print("Reading halos from {0}".format(fname))

    with open(fname, "rb") as f_in:
        NTrees = np.fromfile(f_in, np.dtype(np.int32), 1)[0]
        NHalos = np.fromfile(f_in, np.dtype(np.int32), 1)[0]
        NHalosPerTree = np.fromfile(f_in,
                                    np.dtype((np.int32, NTrees)), 1)[0]

        Halos = np.empty(NHalos, dtype=Halo_Desc)

        # Go through the input file and write those selected trees.
        Halos = np.fromfile(f_in, Halo_Desc, 
                            NHalos)     

    return Halos 


def read_subfind_halos(fname):

    Halo_Desc_full = [
    ('id_MBP',              np.int64),
    ('M_vir',               np.float64),
    ('n_particles',         np.int16),
    ('position_COM',        (np.float32, 3)),
    ('position_MBP',        (np.float32, 3)),
    ('velocity_COM',        (np.float32, 3)),
    ('velocity_MBP',        (np.float32, 3)),
    ('R_vir',               np.float32),
    ('R_halo',              np.float32),
    ('R_max',               np.float32),
    ('V_max',               np.float32),
    ('sigma_v',             np.float32),
    ('spin',                (np.float32, 3)),
    ('q_triaxial',          np.float32),
    ('s_triaxial',          np.float32),
    ('shape_eigen_vectors', (np.float32, (3,3))),
    ('padding',             (np.int16, 2))
                     ] # Note that there are also a padding of 8 bytes following this array. 

    names = [Halo_Desc_full[i][0] for i in range(len(Halo_Desc_full))]
    formats = [Halo_Desc_full[i][1] for i in range(len(Halo_Desc_full))]
    Halo_Desc = np.dtype({'names':names, 'formats':formats}, align=True)

    print("Reading Subfind halos from file {0}".format(fname))
    with open(fname, 'rb') as f_in:
        file_number = np.fromfile(f_in, np.dtype(np.int32), 1)
        n_files = np.fromfile(f_in, np.dtype(np.int32), 1)
        N_groups_thisfile = np.fromfile(f_in, np.dtype(np.int32), 1)[0]
        N_groups_allfile = np.fromfile(f_in, np.dtype(np.int32), 1)
        Halos = np.fromfile(f_in, Halo_Desc, N_groups_thisfile)

    return Halos


def get_num_subfind_halos(fname):

    with open(fname, 'rb') as f_in:
        file_number = np.fromfile(f_in, np.dtype(np.int32), 1)
        n_files = np.fromfile(f_in, np.dtype(np.int32), 1)
        N_groups_thisfile = np.fromfile(f_in, np.dtype(np.int32), 1)[0]
        N_groups_allfile = np.fromfile(f_in, np.dtype(np.int32), 1)[0]

    return N_groups_allfile

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
         ('GridHaloMass', (np.float32, MAXSNAPS)),
         ('EjectedFraction', (np.float32, MAXSNAPS)),  
         ('LenHistory', (np.int32, MAXSNAPS)),
         ('QuasarActivity', (np.int32, MAXSNAPS)),
         ('QuasarSubstep', (np.int32, MAXSNAPS)),
         ('DynamicalTime', (np.float32, MAXSNAPS)),
         ('LenMergerGal', (np.int32, MAXSNAPS)),
         ('GridReionMod', (np.float32, MAXSNAPS)),
         ('GridNgamma_HI', (np.float32, MAXSNAPS)),
         ('Gridfesc', (np.float32, MAXSNAPS))
#         ('GridInfallRate', (np.float32, MAXSNAPS))
         ]

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
    expected_size = GridSize*GridSize*GridSize*byte_size   

    if(expected_size != filesize):
        print("The size of file {0} is {1} bytes whereas we expected it to be "
              "{2} bytes".format(filepath, filesize, expected_size)) 
        raise ValueError("Mismatch between size of file and expected size.")

    fd = open(filepath, 'rb')
    grid = np.fromfile(fd, count = GridSize**3, dtype = readformat) 
    if (reshape == True):
        grid = np.reshape(grid, (GridSize, GridSize, GridSize), order="F") 
    fd.close()

    return grid


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
    Reads the ``SAGE`` ``.ini`` file into a dictionary containing the parameter
    values.

    Parameters
    ----------

    fname : String
        Path to the ``SAGE`` ``.ini`` file.         

    Returns
    ---------

    SAGE_dict : Dictionary
        Dictionary keyed by the ``SAGE`` parameter field names and containing 
        the values from the ``.ini`` file. 

    Errors 
    ---------

    RuntimeError
        Raised if the specified ``SAGE`` ``.ini`` file not found.
    """

    SAGE_fields = ["FileNameGalaxies", "OutputDir", "GridOutputDir",
                   "FirstFile", "LastFile",
                   "TreeName", "TreeExtension", "SimulationDir",
                   "FileWithSnapList", "LastSnapShotNr", "Omega", 
                   "OmegaLambda", "BaryonFrac", "Hubble_h",
                   "PartMass", "BoxSize", "GridSize",
                   "UnitLength_in_cm", "UnitMass_in_g",
                   "UnitVelocity_in_cm_per_s", "self_consistent",
                   "ReionizationOn", "SupernovaRecipeOn",
                   "DiskInstabilityOn", "SFprescription",
                   "AGNrecipeOn", "QuasarRecipeOn",
                   "SfrEfficiency", "FeedbackReheatingEpsilon",
                   "FeedbackEjectionEfficiency", "IRA",
                   "TimeResolutionSN", "ReIncorporationFactor",
                   "RadioModeEfficiency", "QuasarModeEfficiency",
                   "BlackHoleGrowthRate", "ThreshMajorMerger",
                   "ThresholdSatDisruption", "Yield",
                   "RecycleFraction",
                   "Reionization_z0", "Reionization_zr",
                   "EnergySN", "RescaleSN", "IMF",
                   "LowSnap", "HighSnap", "PhotoionDir",
                   "PhotoionName", "ReionRedshiftName",
                   "PhotonPrescription", "HaloPartCut", "TimeResolutionStellar",
                   "fescPrescription", "alpha", "beta", "delta",
                   "quasar_baseline", "quasar_boosted",
                   "N_dyntime", "MH_low", "fesc_low",
                   "MH_high", "fesc_high"
                  ]

    SAGE_dict = {}
 
    try:
        with open (fname, "r") as SAGE_file:
            data = SAGE_file.readlines() 

            for line in range(len(data)):
                stripped = data[line].strip()
                try:
                    first_char = stripped[0]
                except IndexError:
                    continue
                if first_char == ";" or first_char == "%" or first_char == "-": 
                    continue
                split = stripped.split()
                if split[0] in SAGE_fields:
                    SAGE_dict[split[0]] = split[1]

        return SAGE_dict

    except FileNotFoundError:
        print("Could not file SAGE ini file {0}".format(fname))
        raise RuntimeError 


def read_cifog_ini(fname):
    """    
    Reads the ``cifog`` ``.ini`` file into a dictionary containing the parameter
    values.  ``cifog`` also contains a number of header fields which must be
    present in the ``.ini`` file which this function also reads into a separate
    dictionary. 

    Parameters
    ----------

    fname : String
        Path to the ``cifog`` ``.ini`` file.         

    Returns
    ---------

    cifog_dict: Dictionary
        Dictionary keyed by the ``cifog`` parameter field names and containing 
        the values from the ``.ini`` file. 

    cifog_headers: Dictionary
        Dictionary keyed by the ``cifog`` parameter field that comes **after**
        the header.  The value is the header name. 

    Errors 
    ---------

    RuntimeError
        Raised if the specified ``cifog`` ``.ini`` file not found.
    """

    cifog_fields = ["calcIonHistory", "numSnapshots", "stopSnapshot",
                    "redshiftFile", "redshift_prevSnapshot", "finalRedshift",
                    "evolutionTime", "size_linear_scale",
                    "first_increment_in_logscale", "max_scale", 
                    "useDefaultMeanDensity", "useIonizeSphereModel",
                    "useWebModel", "photHImodel", "calcMeanFreePath",
                    "constantRecombinations", "calcRecombinations",
                    "solveForHelium", "paddedBox", "gridsize", "boxsize",
                    "densityFilesAreInDoublePrecision",
                    "nionFilesAreInDoublePrecision", "inputFilesAreComoving",
                    "inputFilesAreSimulation", "SimulationLowSnap",
                    "SimulationHighSnap", "inputIgmDensityFile",
                    "inputIgmDensitySuffix",
                    "densityInOverdensity", "meanDensity", "inputIgmClumpFile",
                    "inputSourcesFile", "inputNionFile", "nion_factor",
                    "output_XHII_file", "write_photHI_file",
                    "output_photHI_file", "output_restart_file", "hubble_h",
                    "omega_b", "omega_m", "omega_l", "sigma8", "Y",
                    "photHI_bg_file", "photHI_bg", "meanFreePathInIonizedMedium",
                    "sourceSlopeIndex", "dnrec_dt", "recombinationTable",
                    "zmin", "zmax", "dz", "fmin", "fmax", "df", "dcellmin",
                    "dcellmax", "ddcell", "inputSourcesHeIFile",
                    "inputNionHeIFile", "inputSourcesHeIIFile",
                    "inputNionHeIIFile", "dnrec_HeI_dt",
                    "dnrec_HeII_dt", "output_XHeII_file",
                    "output_XHeIII_file"
                  ]

    cifog_dict = {}
    cifog_headers = {} 

    try:
        with open (fname, "r") as cifog_file:
            data = cifog_file.readlines() 

            for line in range(len(data)):
                stripped = data[line].strip()
                try:
                    first_char = stripped[0]
                except IndexError:
                    continue

                if first_char == ";" or first_char == "%" or first_char == "-": 
                    continue

                split = stripped.split()

                if split[0] in cifog_fields:
                    cifog_dict[split[0]] = split[2]

                    if header_name:
                        cifog_headers[split[0]] = header_name
                        header_name = None

                if first_char == "[":
                    header_name = stripped

        return cifog_dict, cifog_headers

    except FileNotFoundError:
        print("Could not file cifog ini file {0}".format(fname))
        raise ValueError 
