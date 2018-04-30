#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>

#include "../core_allvars.h"
#include "../core_proto.h"
#include "common.h"
#include "selfcon_grid.h"

// Local Variables //

// Local Proto-Types //

// External Functions //

void update_temporal_array(int p, int halonr, int steps_completed)
{
    int32_t grid_position, step, status;
    int32_t SnapCurr = Halo[halonr].SnapNum;
    // NOTE: We use the Snapshot number of the FOF-Halo (i.e. the main halo the galaxy belongs to) because the snapshot number of the galaxy has been shifted by -1. //
    // This is consistent with the end of the 'evolve_galaxies' function which shifts Gal[p].SnapNum by +1. //
 
    status = determine_1D_idx(Gal[p].Pos[0], Gal[p].Pos[1], Gal[p].Pos[2], &grid_position); 
    if (status == EXIT_FAILURE)
    {
      exit(EXIT_FAILURE);
    }
    
    Gal[p].GridPos = grid_position; 

    Gal[p].GridHistory[SnapCurr] = grid_position; // Remember the grid history of the galaxy over the Snapshot range.
    Gal[p].GridColdGas[SnapCurr] = Gal[p].ColdGas;
    Gal[p].GridHotGas[SnapCurr] = Gal[p].HotGas;
    Gal[p].GridStellarMass[SnapCurr] = Gal[p].StellarMass; // Stellar mass at this snapshot.

    for(step = 0; step < steps_completed; step++) // We loop over the number of steps completed to allow merged galaxies to be updated. 
    {
      Gal[p].GridSFR[SnapCurr] += Gal[p].SfrBulge[step] + Gal[p].SfrDisk[step]; // Star formation rate at this snapshot.
    }

    Gal[p].GridZ[SnapCurr] = get_metallicity(Gal[p].ColdGas, Gal[p].MetalsColdGas); // Metallicity at this snapshot.
    Gal[p].GridFoFMass[SnapCurr] = get_virial_mass(Halo[Gal[p].HaloNr].FirstHaloInFOFgroup); // Virial mass of the central galaxy (i.e. virial mass of the host halo).  
 
    if((Gal[p].EjectedMass < 0.0) || ((Gal[p].HotGas + Gal[p].ColdGas + Gal[p].EjectedMass) == 0.0))
    {
      Gal[p].EjectedFraction[SnapCurr] = 0.0; // Check divide by 0 case.
    }
    else
    { 
        Gal[p].EjectedFraction[SnapCurr] = Gal[p].EjectedMass/(Gal[p].HotGas + Gal[p].ColdGas + Gal[p].EjectedMass);
        // EjectedFraction is the fraction of baryons in the ejected reservoir.
    }
    if (Gal[p].EjectedFraction[SnapCurr] < 0.0 || Gal[p].EjectedFraction[SnapCurr] > 1.0)
    {
      fprintf(stderr, "Found ejected fraction = %.4e \t p = %d \t Gal[p].EjectedMass = %.4e \t Gal[p].HotGas = %.4e \t Gal[p].ColdGas = %.4e\n\n", Gal[p].EjectedFraction[SnapCurr], p, Gal[p].EjectedMass, Gal[p].HotGas, Gal[p].ColdGas); 
      ABORT(EXIT_FAILURE); 
    }

    Gal[p].LenHistory[SnapCurr] = Gal[p].Len;
    if (Gal[p].LenHistory[SnapCurr] < 0)
    {  
      fprintf(stderr, "Have a galaxy with Len < 0.  Galaxy number %d with Len %d.\n", p, Gal[p].Len);
      ABORT(EXIT_FAILURE);
    }
    if (Gal[p].EjectedMass > 0.0)
    {
        Gal[p].GridEjectedMass[SnapCurr] += Gal[p].EjectedMass; 
    }

    Gal[p].DynamicalTime[SnapCurr] = Gal[p].Rvir / Gal[p].Vvir; 
    Gal[p].GridBHMass[SnapCurr] = Gal[p].BlackHoleMass;

    Gal[p].GridDustColdGas[SnapCurr] = Gal[p].DustColdGas;    
    Gal[p].GridDustHotGas[SnapCurr] = Gal[p].DustHotGas;
    Gal[p].GridDustEjectedMass[SnapCurr] = Gal[p].DustEjectedMass;
    Gal[p].GridType[SnapCurr] = Gal[p].Type;

    if (self_consistent == 1 && SnapCurr == HighSnap)
    {
      status = update_selfcon_grid(&Gal[p], grid_position, SnapCurr);
      if (status != EXIT_SUCCESS)
      {
        ABORT(EXIT_FAILURE);
      }
#ifdef DEBUG_SELCON_GRID
      printf("SFR %.4f\tGridPos %d\tNion %.4f\n", Gal[p].GridSFR[SnapCurr], grid_position, SelfConGrid->Nion_HI[grid_position]);
#endif
    }

}

int32_t malloc_temporal_arrays(struct GALAXY *g)
{

#define ALLOCATE_ARRAY_MEMORY(name, length) \
{                                          \
  name = mycalloc(length, sizeof(*(name)));  \
  if (name == NULL)                        \
  {                                        \
    fprintf(stderr, "Out of memory allocating %ld bytes, could not allocate"#name".\n", sizeof(*(name)* length)); \
    return EXIT_FAILURE;                   \
  }                                        \
}

  ALLOCATE_ARRAY_MEMORY(g->GridHistory,         MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridColdGas,         MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridHotGas,         MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridStellarMass,     MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridSFR,             MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridZ,               MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridFoFMass,         MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->EjectedFraction,     MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->LenHistory,          MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->Stars,               SN_Array_Len);
  ALLOCATE_ARRAY_MEMORY(g->GridOutflowRate,     MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridInfallRate,      MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridEjectedMass,     MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->QuasarActivity,      MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->DynamicalTime,       MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->QuasarSubstep,       MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->LenMergerGal,        MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridBHMass,          MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridReionMod,        MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridDustColdGas,     MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridDustHotGas,      MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridDustEjectedMass, MAXSNAPS);
  ALLOCATE_ARRAY_MEMORY(g->GridType,            MAXSNAPS);

  g->IsMalloced = 1; // This way we can check that we're not freeing memory that hasn't been allocated.

  return EXIT_SUCCESS;

#undef ALLOCATE_ARRAY_MEMORY

}

void free_temporal_arrays(struct GALAXY *g)
{
  myfree(g->GridHistory,         sizeof(*(g->GridHistory)) * MAXSNAPS);
  myfree(g->GridColdGas,         sizeof(*(g->GridColdGas)) * MAXSNAPS);
  myfree(g->GridHotGas,         sizeof(*(g->GridColdGas)) * MAXSNAPS);
  myfree(g->GridStellarMass,     sizeof(*(g->GridStellarMass)) * MAXSNAPS);
  myfree(g->GridSFR,             sizeof(*(g->GridSFR)) * MAXSNAPS);
  myfree(g->GridZ,               sizeof(*(g->GridZ)) * MAXSNAPS);
  myfree(g->GridFoFMass,         sizeof(*(g->GridFoFMass)) * MAXSNAPS);
  myfree(g->EjectedFraction,     sizeof(*(g->EjectedFraction)) * MAXSNAPS);
  myfree(g->LenHistory,          sizeof(*(g->LenHistory)) * MAXSNAPS);
  myfree(g->Stars,               sizeof(*(g->Stars)) * SN_Array_Len);
  myfree(g->GridOutflowRate,     sizeof(*(g->GridOutflowRate)) * MAXSNAPS);
  myfree(g->GridInfallRate,      sizeof(*(g->GridInfallRate)) * MAXSNAPS);
  myfree(g->GridEjectedMass,     sizeof(*(g->GridEjectedMass)) * MAXSNAPS);
  myfree(g->QuasarActivity,      sizeof(*(g->QuasarActivity)) * MAXSNAPS);
  myfree(g->DynamicalTime,       sizeof(*(g->DynamicalTime)) * MAXSNAPS);
  myfree(g->QuasarSubstep,       sizeof(*(g->QuasarSubstep)) * MAXSNAPS);
  myfree(g->LenMergerGal,        sizeof(*(g->LenMergerGal)) * MAXSNAPS);
  myfree(g->GridBHMass,          sizeof(*(g->GridBHMass)) * MAXSNAPS);
  myfree(g->GridReionMod,        sizeof(*(g->GridReionMod)) * MAXSNAPS);
  myfree(g->GridDustColdGas,     sizeof(*(g->GridDustColdGas)) * MAXSNAPS);
  myfree(g->GridDustHotGas,      sizeof(*(g->GridDustHotGas)) * MAXSNAPS);
  myfree(g->GridDustEjectedMass, sizeof(*(g->GridDustEjectedMass)) * MAXSNAPS);
  myfree(g->GridType,            sizeof(*(g->GridType)) * MAXSNAPS);

  g->IsMalloced = 0;
}

void write_temporal_arrays(struct GALAXY *g, FILE *fp)
{

#define WRITE_GRID_PROPERTY(name, length)    \
{                                            \
  XASSERT(name != NULL, #name" has a NULL pointer.\n"); \
  nwritten = fwrite(name, sizeof(*(name)), length, fp); \
  XASSERT(nwritten == length, "While writing "#name", we expected to write %d times but wrote %d times instead\n", \
          length, nwritten);                 \
}

#define WRITE_CONVERTED_GRID_PROPERTY(name, length, conversion, type)    \
{                                            \
  XASSERT(name != NULL, #name" has a NULL pointer.\n"); \
  buffer = calloc(length, sizeof(type));\
  XASSERT(buffer != NULL, "Could not allocate memory for a buffer in write_temporal_arrays for "#name".\n"); \
  for (j = 0; j < length; ++j)\
  {\
    ((type*)buffer)[j] = name[j] * conversion;\
  }\
  nwritten = fwrite(buffer, sizeof(type), length, fp); \
  XASSERT(nwritten == length, "While writing"#name", we expected to write %d times but wrote %d times instead\n", \
          length, nwritten);                 \
  free(buffer); \
}

  float SFR_conversion = UnitMass_in_g / UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS / STEPS; 
  int j;  
  int32_t nwritten;
  void *buffer;
 
  nwritten = fwrite(&g->TreeNr, sizeof(g->TreeNr), 1, fp);
 
  XASSERT(g->IsMalloced == 1, "We are trying to write out the grid arrays for a galaxies who has already been freed.\n");
   
  WRITE_GRID_PROPERTY(g->GridHistory, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridColdGas, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridHotGas, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridStellarMass, MAXSNAPS);
  WRITE_CONVERTED_GRID_PROPERTY(g->GridSFR, MAXSNAPS, SFR_conversion, typeof(*(g->GridSFR))); 
  WRITE_GRID_PROPERTY(g->GridZ, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridFoFMass, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->EjectedFraction, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->LenHistory, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridEjectedMass, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->QuasarActivity, MAXSNAPS); 
  WRITE_CONVERTED_GRID_PROPERTY(g->DynamicalTime, MAXSNAPS, UnitTime_in_Megayears, typeof(*(g->DynamicalTime)));
  WRITE_GRID_PROPERTY(g->QuasarSubstep, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->LenMergerGal, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridBHMass, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridReionMod, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridDustColdGas, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridDustHotGas, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridDustEjectedMass, MAXSNAPS);
  WRITE_GRID_PROPERTY(g->GridType, MAXSNAPS);

#undef WRITE_GRID_PROPERTY
#undef WRITE_CONVERTED_GRID_PROPERTY
 
}

// Local Functions //

