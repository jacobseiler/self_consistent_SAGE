================
Saved Properties
================

The following tables lists the properties that are saved in the output files.  
Each output file first contains header information, **printed once**. After 
this header information are the properties for each galaxy.  **Pay attention**,
some properties are temporal properties and will contain values for each
snapshot (e.g., ``GridHistory`` will have an entry for each snapshot) whereas
some properties will only have a single number (e.g., ``TreeNr`` will be a
single integer).


Header
------

+-------------------+--------------------------------------------------------------+---------------+--------------------------------------------------+--------------------+
| **Variable Name** |                 **Definition**                               |  **Data Type** |          **Units/Possible Values**               | **Number Entries** |
+===================+==============================================================+================+==================================================+====================+
| MAXSNAPS          | Number of snapshots in the simulation.                       | 32 bit integer.| Unitless.  Will be greater than 0.               | 1                  |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+
| ZZ                | Redshift of each snapshot.                                   | 32 bit float.  | Unitless.  Will be greater than (or equal to) 0. | MAXSNAPS           |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+
| Hubble_h          | Hubble Parameter of the simulation.                          | 32 bit float.  | Unitless. Will be between 0 and 1                | 1                  |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+
| Omega             | Matter critical density parameter of the simulation.         | 32 bit float.  | Unitless.                                        | 1                  |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+
| OmegaLambda       | Dark energy critical density parameter of the simulation.    | 32 bit float.  | Unitless.                                        | 1                  |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+
| BaryonFrac        | (Cosmic) baryon fraction.                                    | 32 bit float.  | Unitless.                                        | 1                  |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+
| PartMass          | Mass of a single dark matter particle in the simulation.     | 32 bit float.  | Msun/h.                                          | 1                  |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+
| BoxSize           | Side-length of the simulation box.                           | 32 bit float   | Mpc/h.                                           | 1                  |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+
| GridSize          | Number of grid cells along one side.                         | 32 bit float   | Mpc/h.                                           | 1                  |
+-------------------+--------------------------------------------------------------+----------------+--------------------------------------------------+--------------------+

Galaxy Properties
-----------------


