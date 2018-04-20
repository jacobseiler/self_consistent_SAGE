#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "core_allvars.h"
#include "core_proto.h"

#define DOUBLE 1
#define STRING 2
#define INT 3
#define MAXTAGS 300



int32_t read_parameter_file(char *fname)
{
  FILE *fd;
  char buf[400], buf1[400], buf2[400], buf3[400];
  int i, j, nt = 0;
  int id[MAXTAGS];
  void *addr[MAXTAGS];
  char tag[MAXTAGS][50];
  int errorFlag = 0; 


#ifdef MPI
  if(ThisTask == 0)
#endif
    printf("\nreading parameter file:\n\n");

  strcpy(tag[nt], "OutputDir");
  addr[nt] = OutputDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "GridOutputDir");
  addr[nt] = GridOutputDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "FileNameGalaxies");
  addr[nt] = FileNameGalaxies;
  id[nt++] = STRING;

  strcpy(tag[nt], "TreeName");
  addr[nt] = TreeName;
  id[nt++] = STRING;

  strcpy(tag[nt], "TreeExtension");
  addr[nt] = TreeExtension;
  id[nt++] = STRING;


  strcpy(tag[nt], "SimulationDir");
  addr[nt] = SimulationDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "PhotoionDir");
  addr[nt] = PhotoionDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "PhotoionName");
  addr[nt] = PhotoionName;
  id[nt++] = STRING;

  strcpy(tag[nt], "ReionRedshiftName");
  addr[nt] = ReionRedshiftName; 
  id[nt++] = STRING;

  strcpy(tag[nt], "ReionSnap");
  addr[nt] = &ReionSnap; 
  id[nt++] = INT;

  strcpy(tag[nt], "FileWithSnapList");
  addr[nt] = FileWithSnapList;
  id[nt++] = STRING;

  strcpy(tag[nt], "LastSnapShotNr");
  addr[nt] = &LastSnapShotNr;
  id[nt++] = INT;

  strcpy(tag[nt], "FirstFile");
  addr[nt] = &FirstFile;
  id[nt++] = INT;

  strcpy(tag[nt], "LastFile");
  addr[nt] = &LastFile;
  id[nt++] = INT;

  strcpy(tag[nt], "ThreshMajorMerger");
  addr[nt] = &ThreshMajorMerger;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "RecycleFraction");
  addr[nt] = &RecycleFraction;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "ReIncorporationFactor");
  addr[nt] = &ReIncorporationFactor;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "UnitVelocity_in_cm_per_s");
  addr[nt] = &UnitVelocity_in_cm_per_s;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "UnitLength_in_cm");
  addr[nt] = &UnitLength_in_cm;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "UnitMass_in_g");
  addr[nt] = &UnitMass_in_g;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "Hubble_h");
  addr[nt] = &Hubble_h;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "ReionizationOn");
  addr[nt] = &ReionizationOn;
  id[nt++] = INT;

  strcpy(tag[nt], "SupernovaRecipeOn");
  addr[nt] = &SupernovaRecipeOn;
  id[nt++] = INT;

  strcpy(tag[nt], "DiskInstabilityOn");
  addr[nt] = &DiskInstabilityOn;
  id[nt++] = INT;

  strcpy(tag[nt], "AGNrecipeOn");
  addr[nt] = &AGNrecipeOn;
  id[nt++] = INT;

  strcpy(tag[nt], "SFprescription");
  addr[nt] = &SFprescription;
  id[nt++] = INT;

  strcpy(tag[nt], "BHmodel");
  addr[nt] = &BHmodel;
  id[nt++] = INT;

  strcpy(tag[nt], "BaryonFrac");
  addr[nt] = &BaryonFrac;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "Omega");
  addr[nt] = &Omega;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "OmegaLambda");
  addr[nt] = &OmegaLambda;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "PartMass");
  addr[nt] = &PartMass;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "BoxSize");
  addr[nt] = &BoxSize;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "GridSize");
  addr[nt] = &GridSize;
  id[nt++] = INT;

  strcpy(tag[nt], "self_consistent");
  addr[nt] = &self_consistent;
  id[nt++] = INT;

  strcpy(tag[nt], "EnergySN");
  addr[nt] = &EnergySN;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "IRA");
  addr[nt] = &IRA;
  id[nt++] = INT;

  strcpy(tag[nt], "TimeResolutionSN");
  addr[nt] = &TimeResolutionSN;
  id[nt++] = INT;

  strcpy(tag[nt], "IMF");
  addr[nt] = &IMF;
  id[nt++] = INT;

  strcpy(tag[nt], "Yield");
  addr[nt] = &Yield;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "FracZleaveDisk");
  addr[nt] = &FracZleaveDisk;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "SfrEfficiency");
  addr[nt] = &SfrEfficiency;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "FeedbackReheatingEpsilon");
  addr[nt] = &FeedbackReheatingEpsilon;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "FeedbackEjectionEfficiency");
  addr[nt] = &FeedbackEjectionEfficiency;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "BlackHoleGrowthRate");
  addr[nt] = &BlackHoleGrowthRate;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "RadioModeEfficiency");
  addr[nt] = &RadioModeEfficiency;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "QuasarModeEfficiency");
  addr[nt] = &QuasarModeEfficiency;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "Reionization_z0");
  addr[nt] = &Reionization_z0;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "Reionization_zr");
  addr[nt] = &Reionization_zr;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "ThresholdSatDisruption");
  addr[nt] = &ThresholdSatDisruption;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "NumOutputs");
  addr[nt] = &NOUT;
  id[nt++] = INT;

  strcpy(tag[nt], "LowSnap");
  addr[nt] = &LowSnap;
  id[nt++] = INT;

  strcpy(tag[nt], "HighSnap");
  addr[nt] = &HighSnap;
  id[nt++] = INT;

  strcpy(tag[nt], "RescaleSN");
  addr[nt] = &RescaleSN;
  id[nt++] = INT;

  strcpy(tag[nt], "fescPrescription");
  addr[nt] = &fescPrescription;
  id[nt++] = INT;

  strcpy(tag[nt], "alpha");
  addr[nt] = &alpha;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "beta");
  addr[nt] = &beta;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "MH_low");
  addr[nt] = &MH_low;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "fesc_low");
  addr[nt] = &fesc_low;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "MH_high");
  addr[nt] = &MH_high;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "fesc_high");
  addr[nt] = &fesc_high;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "quasar_baseline");
  addr[nt] = &quasar_baseline;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "quasar_boosted");
  addr[nt] = &quasar_boosted;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "N_dyntime");
  addr[nt] = &N_dyntime;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "fesc");
  addr[nt] = &fesc;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "HaloPartCut");
  addr[nt] = &HaloPartCut;
  id[nt++] = INT;

  if((fd = fopen(fname, "r")))
  {
    while(!feof(fd))
    {
      *buf = 0;
      fgets(buf, 200, fd);
      if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
        continue;

      if(buf1[0] == '%' || buf1[0] == '-')
        continue;

      for(i = 0, j = -1; i < nt; i++)
        if(strcmp(buf1, tag[i]) == 0)
      {
        j = i;
        tag[i][0] = 0;
        break;
      }

      if(j >= 0)
      {
#ifdef MPI
        if(ThisTask == 0)
#endif
          printf("%35s\t%10s\n", buf1, buf2);

        switch (id[j])
        {
          case DOUBLE:
          *((double *) addr[j]) = atof(buf2);
          break;
          case STRING:
          strcpy(addr[j], buf2);
          break;
          case INT:
          *((int *) addr[j]) = atoi(buf2);
          break;
        }
      }
      else
      {
        //printf("Error in file %s:   Tag '%s' not allowed or multiple defined.\n", fname, buf1);        
      }
    }
    fclose(fd);

  }
  else
  {
    printf("Parameter file %s not found.\n", fname);
    errorFlag = 1;
  }

  for(i = 0; i < nt; i++)
  {
    if(*tag[i]) 
    {
      if (strcmp(tag[i], "TreeExtension") != 1)
      {
        memset(tag[i], 0, 50); 
      }
      else
      {
        printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], fname);
        errorFlag = 1;
      }
    }
  }
	
  if (errorFlag == 1)
  {
    return EXIT_FAILURE;
  }
	printf("\n");
	
	assert(LastSnapShotNr+1 > 0 && LastSnapShotNr+1 < ABSOLUTEMAXSNAPS);
	MAXSNAPS = LastSnapShotNr + 1;


  XASSERT(NOUT == 1, "The number of outputs must be 1.  The only output will be the galaxies at the final snapshot or the galaxies that have merged before this time.\n");
		
	// read in the output snapshot list
	if(NOUT == -1)
	{
		NOUT = MAXSNAPS;
		for (i=NOUT-1; i>=0; i--)
			ListOutputSnaps[i] = i;
		printf("all %i snapshots selected for output\n", NOUT);
	}
	else
	{
		printf("%i snapshots selected for output: ", NOUT);
		// reopen the parameter file
		fd = fopen(fname, "r");
		
    ListOutputSnaps[0] = LastSnapShotNr;
		fclose(fd);
	
		printf("\n");
	}

  if (GridSize > 1024)
  {
    fprintf(stderr, "The grid size cannot be greater than 1024.\n");
    return EXIT_FAILURE;
  }

  if ((ReionSnap < LowSnap || ReionSnap > HighSnap) && ReionizationOn == 3)
  {
    fprintf(stderr, "The reionization prescription chosen was for the self-consistent run. However the snapshot chosen for the current reionization iteration (snapshot %d) is smaller/larger than the range we are calculating over (LowSnap to SnapHigh, %d to %d).\n", ReionSnap, LowSnap, HighSnap);
    return EXIT_FAILURE;
  }
  

  return EXIT_SUCCESS;

}
