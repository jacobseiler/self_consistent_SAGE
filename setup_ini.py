"""
Script that updates template ``SAGE`` and ``cifog`` ``.ini`` files to use the
specified directories. 

Author: Jacob Seiler
Version: 0.1.
"""

#!/usr/bin/env python
import os
import sys

# Get the directory the script is in.
# Used a global variable for convenience as quite a few functions will use this.
script_dir = os.path.dirname(os.path.realpath(__file__))
output_path = "{0}/output/".format(script_dir)
sys.path.append(output_path)

import ReadScripts as rs

utils_path = "{0}/utils".format(script_dir)
sys.path.append(utils_path)

import change_params as cp


def update_SAGE_dict(base_SAGE_params):
    
    fields_to_check = ["SimulationDir", "FileWithSnapList", "TreeName",
                       "TreeExtension", "FileNameGalaxies"] 
    print_text = ["Path to the halo trees", 
                  "File with the simulation snapshot list",
                  "Prefix for the trees", "Suffix for the trees",
                  "Galaxy prefix name (used as unique run tag)"]
    check_type = ["dir", "file", "", "", "required"]

    updated_fields = update_fields(base_SAGE_params, fields_to_check,
                                   print_text, check_type)

    return updated_fields


def update_cifog_dict(base_cifog_params):

    fields_to_check = ["redshiftFile", "inputIgmDensityFile"]
    print_text = ["cifog redshift file", 
                  "Path to the input density file"]
    check_type = ["file", ""]

    updated_fields = update_fields(base_cifog_params, fields_to_check,
                                   print_text, check_type)

    return updated_fields


def update_fields(base_dict, fields_to_check, print_text, check_type):

    updated_fields = {}

    for idx in range(len(fields_to_check)):

        field = fields_to_check[idx]
        text = print_text[idx]
        check = check_type[idx] 

        if check != "required":

            text_to_print = "{0} [default: {1}]: ".format(text,
                                                          base_dict[field])
            my_field = input("{0}".format(text_to_print))
            if not my_field: 
                my_field = base_dict[field]
        else:
            my_field = None

            text_to_print = "{0} (must be given): ".format(text,
                                                           base_dict[field])
            while not my_field:
                my_field = input("{0}".format(text_to_print))
                if not my_field:
                    print("Must be specified.")

        if check == "dir":
            if not os.path.isdir(my_field):
                print("Path {0} does not exist.".format(my_field))
                raise ValueError
        elif check == "file":
            if not os.path.isfile(my_field):
                print("File {0} does not exist.".format(my_field))
                raise ValueError

        updated_fields[field] = my_field

    return updated_fields


if __name__ == '__main__':

    print("Welcome to the RSAGE ini file adjuster.")
    print("This script will adjust the directory names for both the SAGE and "
          "cifog ini files to be consistent with each other.") 
    print("For any of the following, if nothing is entered, we will use the "
          "default values for the template ini files specified.")
    print("")
    print("")

    base_SAGE_ini = "{0}/ini_files/kali_SAGE.ini".format(script_dir)
    base_cifog_ini = "{0}/ini_files/kali_cifog.ini".format(script_dir)

    # Prompt for a SAGE ini path. If none provided, use the base.
    my_SAGE_ini = input("Template SAGE ini file [default: "
                        "{0}]: ".format(base_SAGE_ini))
    if not my_SAGE_ini:
        my_SAGE_ini = base_SAGE_ini

    # Do the same for cifog.
    my_cifog_ini = input("Template cifog ini file [default: "
                        "{0}]: ".format(base_cifog_ini))
    if not my_cifog_ini:
        my_cifog_ini = base_cifog_ini

    SAGE_params = rs.read_SAGE_ini(my_SAGE_ini)
    cifog_params, cifog_headers = rs.read_cifog_ini(my_cifog_ini)

    run_directory = None
    while not run_directory:
        run_directory = input("Base output directory")
        if not run_directory:
            print("Must be specified")

    SAGE_fields_update = update_SAGE_dict(SAGE_params)
    cifog_fields_update = update_cifog_dict(cifog_params)

    cp.create_directories(run_directory)
    cp.update_ini_files(base_SAGE_ini, base_cifog_ini,
                        SAGE_fields_update, cifog_fields_update,
                        run_directory)