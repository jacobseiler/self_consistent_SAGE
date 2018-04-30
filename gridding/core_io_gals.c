#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>

#include "core_allvars_grid.h"
#include "core_proto_grid.h"

// keep a static file handle to remove the need to do constant seeking
FILE* load_fd = NULL;

int32_t load_gals(char *fname)
{

// Define a macro that will allocate memory for the array properties of galaxies. 
// For some properties the data is read from the input file and otherwise we will initialise it oureselves.
// The `read` variable controls whether data is read from the input file or not. 

#define ALLOCATE_ARRAY_MEMORY(name, length, read) \
{                                                \
  name = calloc(length, sizeof(*(name)));        \
  if (name == NULL)                              \
  {                                              \
    fprintf(stderr, "Out of memory allocating %ld bytes, could not allocate"#name".\n", sizeof(*(name)* length)); \
    return EXIT_FAILURE;                         \
  }                                              \
  if (read == 1)                                 \
    fread(name, sizeof(*(name)), length, infile);\
}

  int32_t i, j;
  FILE *infile;

  infile = fopen(fname, "rb");
  if (infile == NULL) 
  {
    printf("can't open file `%s'\n", fname);
    return EXIT_FAILURE;
  }

#ifdef DEBUG_GALS 
  printf("Loading Galaxies from file %s\n", fname);
#endif
  fread(&Ntrees, 1, sizeof(int), infile);
  fread(&NtotGals, sizeof(int), 1, infile);

  GalsForTree = malloc(Ntrees * sizeof(int));
  fread(GalsForTree, Ntrees, sizeof(int), infile);

  GalGrid = malloc(NtotGals * sizeof(struct GALAXY_GRID));
  
  for (i = 0; i < NtotGals; ++i)
  {

    fread(&GalGrid[i].TreeNr, sizeof(int), 1, infile); 
    fread(&GalGrid[i].mergeType, sizeof(int), 1, infile); 

    // Now want to read in array data from the galaxy files (`read` == 1).
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].History, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].StellarMass, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].SFR, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].Z, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].FoFMass, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].EjectedFraction, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].LenHistory, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].OutflowRate, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].InfallRate, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].EjectedMass, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].QuasarActivity, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].DynamicalTime, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].QuasarSubstep, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].ColdGas, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].LenMergerGal, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].BHMass, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].ReionMod, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].ColdDustMass, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].HotDustMass, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].EjectedDustMass, MAXSNAPS, 1);
    ALLOCATE_ARRAY_MEMORY(GalGrid[i].Type, MAXSNAPS, 1);


    if (GalGrid[i].History[MAXSNAPS - 1] == -1)
    {
      printf("Found a galaxy with -1 GridHistory\n"); 
    }

#ifdef DEBUG_GALS
    if (i == 53)
    {
      int tmp = MAXSNAPS - 1;
      printf("mergeType = %d\tHistory[Final] = %d\tStellarMass = %.4f\tSFR = %.4f\tZ = %.4f\tFoFMass = %.4f\tEjectedFraction = %.4f\tLenHistory = %d\tOutflowRate = %.4f\tInfallRate = %.4f\tEjectedMass = %.4f\tQuasarActivity = %d\tDynamicalTime = %.4f\tQuasarSubstep = %d\tColdGas = %.4f\tLenMergerGal = %d\tBHMass = %.4f\tReionization Modifier = %.4f\n", GalGrid[i].mergeType, GalGrid[i].History[tmp], GalGrid[i].StellarMass[tmp], GalGrid[i].SFR[tmp], GalGrid[i].Z[tmp], GalGrid[i].FoFMass[tmp], GalGrid[i].EjectedFraction[tmp], GalGrid[i].LenHistory[tmp], GalGrid[i].OutflowRate[tmp], GalGrid[i].InfallRate[tmp], GalGrid[i].EjectedMass[tmp], GalGrid[i].QuasarActivity[tmp], GalGrid[i].DynamicalTime[tmp], GalGrid[i].QuasarSubstep[tmp], GalGrid[i].ColdGas[tmp], GalGrid[i].LenMergerGal[tmp], GalGrid[i].BHMass[tmp], GalGrid[i].ReionMod[tmp]);
      fclose(infile);
      return EXIT_FAILURE; 
    }
#endif

  } // Galaxy Loop

  printf("Read in %ld total galaxies.\n", (long)NtotGals);
  exit(EXIT_FAILURE);
  // We will now allocate memory for storing the escape fraction value of each galaxy.

  for (i = 0;i<Grid->NumGrids; ++i)
  {
    ALLOCATE_ARRAY_MEMORY(Grid->GridProperties[i].SnapshotGalaxy, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(Grid->GridProperties[i].fescGalaxy, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(Grid->GridProperties[i].MvirGalaxy, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(Grid->GridProperties[i].MstarGalaxy, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(Grid->GridProperties[i].NgammaGalaxy, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(Grid->GridProperties[i].NgammafescGalaxy, NtotGals, 0);

    for (j = 0; j < NtotGals; ++j)
    {
      Grid->GridProperties[i].SnapshotGalaxy[j] = -1;
      Grid->GridProperties[i].fescGalaxy[j] = 0.0;
      Grid->GridProperties[i].MvirGalaxy[j] = 0.0;
      Grid->GridProperties[i].MstarGalaxy[j] = 0.0;
      Grid->GridProperties[i].NgammaGalaxy[j] = 0.0;
      Grid->GridProperties[i].NgammafescGalaxy[j] = 0.0;
    }
  }
    
  // We now need to allocate memory for the quasar boosted escape fraction (if that prescription is selected).
  if (fescPrescription == 4) 
  {
 
    ALLOCATE_ARRAY_MEMORY(QuasarActivityToggle, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(QuasarActivitySubstep, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(QuasarSnapshot, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(TargetQuasarTime, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(QuasarBoostActiveTime, NtotGals, 0);
    ALLOCATE_ARRAY_MEMORY(QuasarFractionalPhoton, NtotGals, 0);

    for (i = 0; i < NtotGals; ++i)
    {
      QuasarActivityToggle[i] = 0.0;
      QuasarActivitySubstep[i] = -1;
      QuasarSnapshot[i] = -1;
      TargetQuasarTime[i] = 0.0;
      QuasarBoostActiveTime[i] = 0.0;
      QuasarFractionalPhoton[i] = 0.0; 
    }

    printf("Quasar Tracking stuff allocated.\n");

  } // fesc if condition.

  fclose(infile);
  free(GalsForTree);

  return EXIT_SUCCESS;
}

void free_gals(void)
{

  int i;
  for (i = 0; i < NtotGals; ++i)
  {     
    free(GalGrid[i].History);
    free(GalGrid[i].StellarMass);
    free(GalGrid[i].SFR);
    free(GalGrid[i].Z);
    free(GalGrid[i].FoFMass); 
    free(GalGrid[i].EjectedFraction);
    free(GalGrid[i].LenHistory);
    free(GalGrid[i].OutflowRate); 
    free(GalGrid[i].InfallRate); 
    free(GalGrid[i].EjectedMass); 
    free(GalGrid[i].QuasarActivity); 
    free(GalGrid[i].DynamicalTime);
    free(GalGrid[i].QuasarSubstep); 
    free(GalGrid[i].ColdGas); 
    free(GalGrid[i].LenMergerGal); 
    free(GalGrid[i].BHMass);
    free(GalGrid[i].ReionMod); 
  }

  for (i = 0; i < Grid->NumGrids; ++i)
  {
    free(Grid->GridProperties[i].SnapshotGalaxy);
    free(Grid->GridProperties[i].fescGalaxy);
    free(Grid->GridProperties[i].MvirGalaxy);      
    free(Grid->GridProperties[i].MstarGalaxy);      
    free(Grid->GridProperties[i].NgammaGalaxy);      
    free(Grid->GridProperties[i].NgammafescGalaxy);      
  }
  
  if (fescPrescription == 4)
  {
    free(QuasarActivityToggle);
    free(QuasarActivitySubstep);
    free(QuasarSnapshot); 
    free(TargetQuasarTime);
    free(QuasarBoostActiveTime);
    free(QuasarFractionalPhoton);
  }

  free(GalGrid);
}

