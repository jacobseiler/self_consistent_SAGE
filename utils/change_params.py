"""
Creates directories and ``.ini`` files for ``RSAGE`` runs. By specifying lists of
variables in ``SAGE_fields_update`` and ``cifog_fields_update``, you are able
to create a unique combination of variables to update for each run. 

The script also gives the option of making ``slurm`` files with the paths
correctly specified and also submit them (**BE CAREFUL WHEN SUBMITTING**).

All ``.ini`` and ``.slurm`` files are created using template files.  Examples
files are included in the base repo.

Author: Jacob Seiler
Version: 0.1.
"""

#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys
import os
from shutil import copyfile
import subprocess

# Get the directory the script is in.
# Used a global variable for convenience as quite a few functions will use this.
script_dir = os.path.dirname(os.path.realpath(__file__))
output_path = "{0}/../output/".format(script_dir)
sys.path.append(output_path)

import ReadScripts
import AllVars


def check_input_parameters(SAGE_fields_update, cifog_fields_update,
                           run_directories, base_SAGE_ini, base_cifog_ini,
                           base_slurm_file):
    """
    Checks that the update fields all have the same number of inputs.  Also
    checks that the template ``.ini`` and ``.slurm`` files exist. 
 
    Parameters
    ----------

    SAGE_fields_update, cifog_fields_update : Dictionaries
        Fields that will be updated and their new value.

    run_directories : List of strings, length equal to number of runs 
        Path to the base ``RSAGE`` directory for each run where all the model
        output will be placed.

    base_SAGE_ini, base_cifog_ini : Strings
        Paths to the template SAGE and cifog ``.ini`` files.

    base_slurm_file : String
        Path to the template ``.slurm`` file.

    Returns
    ----------
    
    None.

    Errors 
    ----------

    ValueError:
        Raised if any of the fields in ``SAGE_fields_update`` or
        ``cifog_fields_update`` have different lengths.

        Raised if any of the fields in ``SAGE_fields_update`` or
        ``cifog_fields_update`` have different lengths to that of
        ``run_directories``.

        Raised in any of the template files do not exist.    
    """

    # Compare every field within ``SAGE_fields_update`` and
    # ``cifog_fields_update`` to ensure they have the same lengths.
    for count, my_dict in enumerate([SAGE_fields_update, cifog_fields_update]):
        if count == 0:
            which_dict = "SAGE"
        else:
            which_dict = "cifog"
        for key_1 in SAGE_fields_update.keys(): 
            for key_2 in SAGE_fields_update.keys():
 
                if len(SAGE_fields_update[key_1]) != len(SAGE_fields_update[key_2]):
 
                    print("For the {0} update dictionary, Key {1} did not have "
                          "the same length as key {2}".format(which_dict,
                                                              key_1, 
                                                              key_2))
                    raise ValueError

            if len(SAGE_fields_update[key_1]) != len(run_directories):
                print("For the {0} update dictionary, Key {1} did not have "
                      "the same length as the number of run directories" \
                      .format(which_dict, key_1))
                raise ValueError

    # Check all the template files exist.
    for my_file in [base_SAGE_ini, base_cifog_ini, base_slurm_file]: 
        if not os.path.isfile(my_file):

            print("File {0} does not exist.".format(my_file))
            raise ValueError


def create_directories(run_directory):
    """
    Creates the directories to house all the ``RSAGE`` outputs.   
 
    Parameters
    ----------

    run_directory: String 
        Path to the base ``RSAGE`` directory. 

    Returns
    ----------
    
    None.
    """

    # First create the base directory.
    base_dir = run_directory 
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print("Created directory {0}".format(base_dir))

    # Directory that will contain all the SAGE output.
    gal_dir = "{0}/galaxies".format(base_dir)
    if not os.path.exists(gal_dir):
        os.makedirs(gal_dir)
        print("Created directory {0}".format(gal_dir))

    # Directory that will contain all the grids. 
    grids_dir = "{0}/grids".format(base_dir)
    if not os.path.exists(grids_dir):
        os.makedirs(grids_dir)
        print("Created directory {0}".format(grids_dir))
       
        # If the grids directory didn't exist, there's no way these will.

        dirs = ["grids/nion", "grids/cifog",
                "grids/cifog/reionization_modifiers",
                "ini_files", "slurm_files", "log_files"]
        for directory in dirs:
            dir_ = "{0}/{1}".format(base_dir, directory)
            os.makedirs(dir_)
            print("Created directory {0}".format(dir_))

   
def update_ini_files(base_SAGE_ini, base_cifog_ini,
                     SAGE_fields_update, cifog_fields_update,
                     run_directory):
    """
    Using template ini files for ``SAGE`` and ``cifog``, creates new ones with 
    the directory paths and field names updated. 

    Parameters
    ----------

    base_SAGE_ini, base_cifog_ini : Strings
        Paths to the template SAGE and cifog ini files.

    SAGE_fields_update, cifog_fields_update : Dictionaries
        Fields that will be updated and their new value.

    run_directory : String
        Path to the base ``RSAGE`` directory.

    Returns
    ----------

    SAGE_fname, cifog_fname : Strings
        Names of the newly created ``SAGE`` and ``cifog`` ini files.
    """

    SAGE_params = ReadScripts.read_SAGE_ini(base_SAGE_ini)
    cifog_params, cifog_headers = ReadScripts.read_cifog_ini(base_cifog_ini)

    # This is the outermost directory. 
    SAGE_params["OutputDir"] = "{0}".format(run_directory)

    # Within RSAGE, we use values of `None` to signify that RSAGE should
    # determine the paths on runtime. 
    SAGE_params["GalaxyOutputDir"] = "None"
    SAGE_params["GridOutputDir"] = "None"
    SAGE_params["PhotoionDir"] = "None"
    SAGE_params["PhotoionName"] = "None"
    SAGE_params["ReionRedshiftName"] = "None"

    cifog_params["inputNionFile"] = "None"
    cifog_params["output_XHII_file"] = "None"

    cifog_params["output_photHI_file"] = "None"
    cifog_params["output_restart_file"] = "None"

    # Now go through the parameters and update them.
    for name in SAGE_fields_update:
        SAGE_params[name] = SAGE_fields_update[name] 

    for name in cifog_fields_update:
        cifog_params[name] = cifog_fields_update[name] 

    # The unique identifier amongst each run will be `FileNameGalaxies`. 
    prefix_tag = SAGE_params["FileNameGalaxies"]

    # Write out the new ini files, using `FileNameGalaxies` as the tag.
    SAGE_fname = "{0}/ini_files/{1}_SAGE.ini".format(run_directory,
                                                     prefix_tag) 

    cifog_fname = "{0}/ini_files/{1}_cifog.ini".format(run_directory,
                                                       prefix_tag) 

    with open (SAGE_fname, "w+") as f:
        for name in SAGE_params.keys():
            string = "{0} {1}\n".format(name, SAGE_params[name])
            f.write(string)

    with open (cifog_fname, "w+") as f:
        for name in cifog_params.keys():

            if name in cifog_headers.keys():
                string = "{0}\n".format(cifog_headers[name])
                f.write(string)

            string = "{0} = {1}\n".format(name, cifog_params[name])
            f.write(string)

    return SAGE_fname, cifog_fname


def make_ini_files(base_SAGE_ini, base_cifog_ini, 
                   SAGE_fields_update, cifog_fields_update,
                   run_directories):
    """
    Makes new ``SAGE`` and ``cifog`` ``.ini`` files for each run. 

    Parameters
    ----------

    base_SAGE_ini, base_cifog_ini : String
        Paths to the template ``SAGE`` and ``cifog`` ``.ini`` files.

    SAGE_fields_update, cifog_fields_update : Dictionaries of lists
        The ``SAGE`` and ``cifog`` parameter fields that will be updated for
        each run.  The values for each parameter field key are length Nx1 where
        N is the number of runs. 
        
    run_directories : List of strings, length equal to number of runs 
        Path to the base ``RSAGE`` directory for each run where all the model
        output will be placed.
 
    Returns
    ----------

    SAGE_ini_names, cifog_ini_names : List of strings, length equal to number
                                      of runs
        Paths to the ini file created for each run.
    """

    SAGE_ini_names = []
    cifog_ini_names = []

    # Now for each run, create a unique dictionary containing the fields for 
    # this run, update the ini files then create all the output directories. 
    for run_number in range(len(run_directories)):

        create_directories(run_directories[run_number])

        thisrun_SAGE_update = {}
        for name in SAGE_fields_update.keys():
            thisrun_SAGE_update[name] = SAGE_fields_update[name][run_number]

        thisrun_cifog_update = {}
        for name in cifog_fields_update.keys():
            thisrun_cifog_update[name] = cifog_fields_update[name][run_number]

        SAGE_fname, cifog_fname = update_ini_files(base_SAGE_ini, base_cifog_ini,
                                                   thisrun_SAGE_update, thisrun_cifog_update,
                                                   run_directories[run_number])        

        SAGE_ini_names.append(SAGE_fname)
        cifog_ini_names.append(cifog_fname)

    return SAGE_ini_names, cifog_ini_names


def make_slurm_files(base_slurm_file, SAGE_ini_names, cifog_ini_names, 
                     run_directories, Nproc): 
    """
    Makes ``slurm`` files for each run. 

    Parameters
    ----------

    base_slurm_file : String
        Path to the template slurm file.

    SAGE_ini_names, cifog_ini_names : List of strings, length equal to number
                                      of runs
        Paths to the ini file created for each run.

    run_directories : List of strings, length equal to number of runs 
        Path to the base ``RSAGE`` directory for each run where all the model
        output will be placed.

    Nproc : Integer
        Number of processors that each run will be executed with.
 
    Returns
    ----------

    slurm_names : List of strings, length equal to number
                                      of runs
        Paths to the ``slurm`` file created for each run.
    """
    slurm_names = []

    for run_number in range(len(SAGE_ini_names)):
        
        SAGE_params = ReadScripts.read_SAGE_ini(SAGE_ini_names[run_number])
        run_name = SAGE_params["FileNameGalaxies"]

        slurm_fname = "{0}/slurm_files/{1}.slurm".format(run_directories[run_number],
                                                         run_name) 

        tmp_slurm_fname = "{0}.tmp".format(base_slurm_file)
        copyfile(base_slurm_file, tmp_slurm_fname)

        # Want to replace lines in the slurm file. Set up the strings. 
        job_name = "#SBATCH --job-name={0}".format(run_name) 
        ntask = "#SBATCH --ntasks={0}".format(Nproc)
        NUMPROC = "NUMPROC={0}".format(Nproc)
        SAGE_ini = 'SAGE_ini="{0}"'.format(SAGE_ini_names[run_number])
        cifog_ini = 'cifog_ini="{0}"'.format(cifog_ini_names[run_number])
        run_prefix = 'run_prefix="{0}"'.format(run_name) 
        path_to_log = 'path_to_log="{0}/log_files/{1}.log"'.format(run_directories[run_number], run_name)

        # Replace strings at specific line numbers.
        line_numbers = [2, 4, 17, 19, 20, 24, 25]  
        string_names = [job_name, ntask, NUMPROC, SAGE_ini, cifog_ini, 
                        run_prefix, path_to_log]
        for line, name in zip(line_numbers, string_names):
            # Use @ as the delimiter here for sed. 
            command = "sed -i '{0}s@.*@{1}@' {2} ".format(line, name,
                                                          tmp_slurm_fname)
            subprocess.call(command, shell=True)

        # Finally move the temporary file to the final location.
        command = "mv {0} {1}".format(tmp_slurm_fname, slurm_fname)
        subprocess.call(command, shell=True)
        print("Created {0}".format(slurm_fname))

        slurm_names.append(slurm_fname)

    return slurm_names


def submit_slurm_jobs(slurm_names):
    """
    Submits the ``slurm`` jobs for each run. 

    Parameters
    ----------

    slurm_names : List of strings, length equal to number
                                      of runs
        Paths to the ``slurm`` file created for each run.
 
    Returns
    ----------

    None.
    """

    for slurm_fname in slurm_names:

        command = "sbatch {0}".format(slurm_fname)        
        subprocess.call(command, shell=True)
         

def create_and_submit(SAGE_fields_update, cifog_fields_update, run_directories,
                      base_SAGE_ini, base_cifog_ini, base_slurm_file, Nproc,
                      submit_slurm=0): 
    """
    Creates all the new ini files and submits them to the queue (if requested).

    Parameters
    ----------

    SAGE_fields_update, cifog_fields_update : Dictionaries
        Fields that will be updated and their new value.

    run_directories : List of strings, length equal to number of runs 
        Path to the base ``RSAGE`` directory for each run where all the model
        output will be placed.

    base_SAGE_ini, base_cifog_ini : Strings
        Paths to the template SAGE and cifog ``.ini`` files.

    base_slurm_file : String
        Path to the template ``.slurm`` file.

    Nproc : Integer.
        Number of processors the code will be run on. 

    submit_slurm : Integer, optional.
        Flag to denote whether we're submitting the jobs to the PBS queue.

        .. note::
            Be very careful with this. If you have (e.g.,) 8 different
            parameters, this will submit 8 jobs. 

    Returns
    ----------

    None.
    """

    # First ensure that the lengths of the input arrays are all in order.
    check_input_parameters(SAGE_fields_update, cifog_fields_update,
                           run_directories, base_SAGE_ini, base_cifog_ini,
                           base_slurm_file)

    # Then generate the new ini files.
    SAGE_ini_names, cifog_ini_names = make_ini_files(base_SAGE_ini, base_cifog_ini, 
                                                     SAGE_fields_update, cifog_fields_update,
                                                     run_directories)

    # Make the slurm files that correspond to these files.
    slurm_names = make_slurm_files(base_slurm_file, SAGE_ini_names, 
                                   cifog_ini_names, run_directories, Nproc)

    # If we're submitting, submit em!
    if submit_slurm:
        submit_slurm_jobs(slurm_names)

 
if __name__ == '__main__':

    # ===================================================================== #
    # Specify here the SAGE parameters that you want to change for each run #
    # ===================================================================== #
    fescPrescription = [1, 1, 1, 1]
    alpha = [0.23, 0.25, 0.27, 0.28] 
    beta = [0.10, 0.10, 0.10, 0.10]
    #delta = [1.00, 1.00, 1.00]
    #MH_low = [1e8, 1e8, 1e8, 1e8, 1e8]
    #fesc_low = [0.01, 0.01, 0.01, 0.01, 0.01]
    #MH_high = [5.0e11, 1.0e12, 1.0e12, 5.0e11, 1.0e11]
    #fesc_high = [0.50, 0.45, 0.40, 0.40, 0.40]

    FileNameGalaxies = ["fej_alpha0.23_beta0.10",
                        "fej_alpha0.25_beta0.10",
                        "fej_alpha0.27_beta0.10",
                        "fej_alpha0.28_beta0.10"]

    SAGE_fields_update = { "fescPrescription" : fescPrescription,
                           #"alpha" : alpha,
                           "beta" : beta,
                           #"delta" : delta,
                           #"MH_low" : MH_low,
                           #"fesc_low" : fesc_low,
                           #"MH_high" : MH_high,
                           #"fesc_high" : fesc_high,
                           "FileNameGalaxies" : FileNameGalaxies
                         }

    # ====================================================================== #
    # Specify here the cifog parameters that you want to change for each run #
    # ====================================================================== #
    cifog_fields_update = {}

    # ============================================ #
    # Specify here the path directory for each run #
    # ============================================ #
    run_directories = ["/fred/oz004/jseiler/kali/self_consistent_output/rsage_fej",
                       "/fred/oz004/jseiler/kali/self_consistent_output/rsage_fej",
                       "/fred/oz004/jseiler/kali/self_consistent_output/rsage_fej",
                       "/fred/oz004/jseiler/kali/self_consistent_output/rsage_fej"]

    # ===================================================================== #
    # Specify here the path to the base ini files (shouldn't need to touch) #  
    # ===================================================================== #
    base_SAGE_ini = "{0}/../ini_files/kali_SAGE.ini".format(script_dir)
    base_cifog_ini = "{0}/../ini_files/kali_cifog.ini".format(script_dir)
    base_slurm_file = "{0}/../run_rsage.slurm".format(script_dir)

    # Misc. 
    Nproc = 32  # Number of processors to run on.

    # CAREFUL CAREFUL.
    # Setting this to 1 will submit the slurm scripts to the PBS queue.
    submit_slurm = 1

    # Now run!
    create_and_submit(SAGE_fields_update, cifog_fields_update, run_directories,
                      base_SAGE_ini, base_cifog_ini, base_slurm_file, Nproc,
                      submit_slurm) 
