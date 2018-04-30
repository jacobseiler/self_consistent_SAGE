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

eroqjeo
  
+-------------------+----------------+---------------+---------------------------+--------------------+
| **Variable Name** | **Definition** | **Data Type** | **Units/Possible Values** | **Number Entries** |
+===================+================+===============+===========================+====================+
|      MAXSNAPS     | The number of snapshots in the simulation | 32 bit integer | Unitless.  Will be greater than 0. | 1 |
+-------------------+----------------+---------------+---------------------------+--------------------+


