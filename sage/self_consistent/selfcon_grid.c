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
#include "selfcon_grid.h"

#ifdef MPI
#include <mpi.h>
struct SELFCON_GRID_STRUCT *MPI_sum_grids(void);
#endif
// Local Variables //

// Local Proto-Types //

void determine_fesc_constants(void);
int32_t malloc_selfcon_grid(struct SELFCON_GRID_STRUCT *grid);
int32_t write_selfcon_grid(struct SELFCON_GRID_STRUCT *grid_towrite);

// External Functions //

int32_t init_selfcon_grid(void)
{

  int32_t status;

#ifdef MPI
  if (ThisTask == 0)
#endif
  printf("Initializing the grids for the self_consistent run.\n");

  if (fescPrescription == 0)
  {
#ifdef MPI
    if (ThisTask == 0)
#endif
    printf("\n\nUsing a constant escape fraction of %.4f\n", fesc); 
  }
  else if (fescPrescription == 1)
  {
    fprintf(stderr, "\n\nDeprecated! Use a different value for fescPrescription one!\n");
    return EXIT_FAILURE;
  } 
  else if (fescPrescription == 2)
  {
#ifdef MPI
    if (ThisTask == 0)
#endif
    printf("\n\nUsing an fesc prescription that scales with halo mass.\n\n");
    determine_fesc_constants();
  }
  else if (fescPrescription == 3)
  {
#ifdef MPI
    if (ThisTask == 0)
#endif
    printf("\n\nUsing an fesc prescription that scales with the fraction of ejected mass in the galaxy.\nThis takes the form A*fej + B with A = %.4e and B = %.4e\n", alpha, beta); 
  }
  else if (fescPrescription == 4)
  {
#ifdef MPI
    if (ThisTask == 0)
#endif
    {
      printf("\n\nUsing an fesc prescription that depends upon quasar activity.\n");
      printf("\nFor a galaxy that had a quasar event within %.2f dynamical times go the escape fraction will be %.2f.  Otherwise it will have a constant value of %.2f\n", N_dyntime, quasar_boosted, quasar_baseline);
    }
  }
  else if (fescPrescription == 5)
  {    
#ifdef MPI
    if (ThisTask == 0)
#endif
    printf("\n\nUsing Anne's functional form for an escape fraction that decreases for increasing halo mass.\n");
    XASSERT(fesc_low > fesc_high, "Input file contain fesc_low = %.2f and fesc_high = %.2f. For this prescription (fescPrescription == 5), we require fesc_low > fesc_high\n", fesc_low, fesc_high);
  }
  else if (fescPrescription == 6)
  {
#ifdef MPI
    if (ThisTask == 0)
#endif
    printf("\n\nUsing Anne's functional form for an escape fraction that increases for increasing halo mass.\n");
    XASSERT(fesc_low < fesc_high, "Input file contain fesc_low = %.2f and fesc_high = %.2f. For this prescription (fescPrescription == 6), we require fesc_low < fesc_high\n", fesc_low, fesc_high);
  }
  else
  {
    printf("\n\nOnly escape fraction prescriptions 0 to 6 (exlucding 1) are permitted.\n");
    return EXIT_FAILURE;
  }


  SelfConGrid = malloc(sizeof(*(SelfConGrid)));
  if (SelfConGrid == NULL)
  {
    fprintf(stderr, "Could not allocate memory for the high level self_consistent grid structure\n");
    return EXIT_FAILURE;
  }

  status = malloc_selfcon_grid(SelfConGrid);
  if (status != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;

}

int32_t free_selfcon_grid(struct SELFCON_GRID_STRUCT *grid_to_free)
{ 

  myfree(grid_to_free->Nion_HI, sizeof(*(grid_to_free->Nion_HI)) * grid_to_free->NumCellsTotal);
  myfree(grid_to_free->GalCount, sizeof(*(grid_to_free->GalCount)) * grid_to_free->NumCellsTotal);
  
  free(grid_to_free);

  return EXIT_SUCCESS;

}

int32_t update_selfcon_grid(struct GALAXY *g, int32_t grid_idx, int32_t snapshot)
{

  int32_t status;
  float Ngamma_HI, Ngamma_HeI, Ngamma_HeII, fesc_local;
  float SFR_conversion = UnitMass_in_g / UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS / STEPS; 

  status = determine_nion(g->GridSFR[snapshot] * SFR_conversion, g->GridZ[snapshot], &Ngamma_HI, &Ngamma_HeI, &Ngamma_HeII);
  if (status != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  status = determine_fesc(g, snapshot, &fesc_local);
  if (status != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (Ngamma_HI > 0.0)
  {
    SelfConGrid->Nion_HI[grid_idx] += pow(10, Ngamma_HI - 50.0) * fesc_local; // Keep the number of ionizing photons in units of 10^50 photons/s. 
  }

  ++(SelfConGrid->GalCount[grid_idx]);

  return EXIT_SUCCESS;
}

int32_t save_selfcon_grid()
{

  int32_t status;

#ifdef MPI

  struct SELFCON_GRID_STRUCT *master_grid;

  master_grid = MPI_sum_grids(); 
  if (ThisTask == 0 && master_grid == NULL)
  {
    return EXIT_FAILURE;
  }

  if (ThisTask == 0)
  {
    status = write_selfcon_grid(master_grid);
  }
  else
    status = EXIT_SUCCESS;

#else
  status = write_selfcon_grid(SelfConGrid);
#endif

  if (status != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }

#ifdef MPI
  if (ThisTask == 0)
    free_selfcon_grid(master_grid);    
#endif

  return EXIT_SUCCESS;

}

int32_t determine_nion(float SFR, float Z, float *Ngamma_HI, float *Ngamma_HeI, float *Ngamma_HeII)
{

  if (SFR == 0)
  {
    *Ngamma_HI = 0;
    *Ngamma_HeI = 0;
    *Ngamma_HeII = 0;
  }
  else if (Z < 0.0025) // 11
  {
    *Ngamma_HI = log10(SFR) + 53.354;
    *Ngamma_HeI = log10(SFR) + 52.727;
    *Ngamma_HeII = log10(SFR) + 48.941;
  }
  else if (Z >= 0.0025 && Z < 0.006) // 12
  {
    *Ngamma_HI = log10(SFR) + 53.290;
    *Ngamma_HeI = log10(SFR) + 52.583;
    *Ngamma_HeII = log10(SFR) + 49.411;
  }
  else if (Z>= 0.006 && Z < 0.014) // 13
  {
    *Ngamma_HI = log10(SFR) + 53.248;
    *Ngamma_HeI = log10(SFR) + 52.481;
    *Ngamma_HeII = log10(SFR) + 49.254;
  }
  else if (Z >= 0.014 && Z < 0.030) // 14
  {
    *Ngamma_HI = log10(SFR) + 53.166;
    *Ngamma_HeI = log10(SFR) + 52.319;
    *Ngamma_HeII = log10(SFR) + 48.596;
  }
  else // 15
  {
    *Ngamma_HI = log10(SFR) + 53.041;
    *Ngamma_HeI = log10(SFR) + 52.052;
    *Ngamma_HeII = log10(SFR) + 47.939;
  }

  if (SFR != 0)
  {
    assert(*Ngamma_HI > 0.0);
    assert(*Ngamma_HeI > 0.0);
    assert(*Ngamma_HeII > 0.0);
  }

  return EXIT_SUCCESS;

}

int32_t determine_fesc(struct GALAXY *g, int32_t snapshot, float *fesc_local)
{

  float halomass = g->GridFoFMass[snapshot] * 1.0e10 / Hubble_h;
  float ejectedfraction = g->EjectedFraction[snapshot];

  switch(fescPrescription)
  {
    case 0:
      *fesc_local = fesc;
      break;

    case 1:
      *fesc_local = pow(10,1.0 - 0.2*log10(halomass)); // Deprecated.
      break;

    case 2:
      *fesc_local = alpha * pow((halomass), beta);
      break;

    case 3:
      *fesc_local = alpha * ejectedfraction + beta;
      break;

    /*
    case 4:
      *fesc_local = quasar_baseline * (1 - QuasarFractionalPhoton[p])  + quasar_boosted * QuasarFractionalPhoton[p];
      break;
    */
    case 5:
      *fesc_local = pow(fesc_low * (fesc_low/fesc_high),(-log10(halomass/MH_low)/log10(MH_high/MH_low)));
      if (*fesc_local > fesc_low)
      {
        *fesc_local = fesc_low;
      }
      break;

    case 6:
      *fesc_local = 1. - pow((1.-fesc_low) * ((1.-fesc_low)/(1.-fesc_high)),(-log10(halomass/MH_low)/log10(MH_high/MH_low)));
      if (*fesc_local < fesc_low)
      {
        *fesc_local = fesc_low;
      }
      break;

    default:
      fprintf(stderr, "The selected fescPrescription is not handled by the switch case in `determine_fesc` in `selfcon_grid.c`.\nPlease add it there.\n");
      return EXIT_FAILURE;

  }

  if (*fesc_local > 1.0 || *fesc_local < 0.0)
  {
    fprintf(stderr, "Had fesc_local = %.4f with halo mass %.4e (log Msun), Stellar Mass %.4e (log Msun), SFR %.4e (log Msun yr^-1) and Ejected Fraction %.4e\n", *fesc_local, log10(halomass * 1.0e10 / Hubble_h), log10(halomass * 1.0e10 / Hubble_h), log10(g->GridSFR[snapshot]), ejectedfraction);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


// Local Functions //

void determine_fesc_constants(void)
{ 
  
  double A, B, log_A;
  
  if (fescPrescription == 2)
  { 
    log_A = (log10(fesc_high) - (log10(fesc_low)*log10(MH_high)/log10(MH_low))) * pow(1 - (log10(MH_high) / log10(MH_low)), -1);
    B = (log10(fesc_low) - log_A) / log10(MH_low);
    A = pow(10, log_A);
    
    alpha = A;
    beta = B;
    
    printf("Fixing the points (%.4e, %.2f) and (%.4e, %.2f)\n", MH_low, fesc_low, MH_high, fesc_high);
    printf("This gives a power law with constants A = %.4e, B = %.4e\n", alpha, beta);
  }
 
}

int32_t malloc_selfcon_grid(struct SELFCON_GRID_STRUCT *my_grid)
{

#define ALLOCATE_GRID_MEMORY(name, length) \
{                                          \
  name = mycalloc(length, sizeof(*(name)));  \
  if (name == NULL)                        \
  {                                        \
    fprintf(stderr, "Out of memory allocating %ld bytes, could not allocate"#name".\n", sizeof(*(name)* length)); \
    return EXIT_FAILURE;                   \
  }                                        \
}

  int32_t cell_idx;

  if (my_grid == NULL)
  {
    fprintf(stderr, "`init_selfcon_grid` was called with a SELFCON_GRID_STRUCT pointer that has not been initialized\n");
    return EXIT_FAILURE;
  }

  my_grid->GridSize = GridSize;
  my_grid->NumCellsTotal = CUBE(GridSize);

  ALLOCATE_GRID_MEMORY(my_grid->Nion_HI, my_grid->NumCellsTotal);
  ALLOCATE_GRID_MEMORY(my_grid->GalCount, my_grid->NumCellsTotal);

  for (cell_idx = 0; cell_idx < my_grid->NumCellsTotal; ++cell_idx)
  {
    my_grid->Nion_HI[cell_idx] = 0.0;
    my_grid->GalCount[cell_idx] = 0;
  }

  return EXIT_SUCCESS;

#undef ALLOCATE_GRID_MEMORY

}

#ifdef MPI

struct SELFCON_GRID_STRUCT *MPI_sum_grids(void)
{

  int32_t status;
  struct SELFCON_GRID_STRUCT *master_grid;
  master_grid = malloc(sizeof(*(master_grid))); // Needs to be malloced for all Tasks for Reduce.

  if (ThisTask == 0)
  {
  
    status = malloc_selfcon_grid(master_grid);
    if (status != EXIT_SUCCESS)
    {
      fprintf(stderr, "Could not allocate memory when trying to sum the grids across tasks.\n");
      return NULL;
    } 
    printf("Reducing selfcon grid.\n");
  }

  MPI_Reduce(SelfConGrid->Nion_HI, master_grid->Nion_HI, SelfConGrid->NumCellsTotal, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (ThisTask == 0)
    return master_grid;
  else
    return NULL; 

}

#endif

int32_t write_selfcon_grid(struct SELFCON_GRID_STRUCT *grid_towrite)
{ 

  FILE* file_HI;
  char tag[MAX_STRING_LEN], fname_HI[MAX_STRING_LEN];
  int32_t nwritten;

  switch(fescPrescription)
  {
    case 0:
      snprintf(tag, MAX_STRING_LEN - 1, "fesc%.2f_HaloPartCut%d", fesc, HaloPartCut);
      break;

    case 1:
      return EXIT_FAILURE; 

    case 2:
      snprintf(tag, MAX_STRING_LEN - 1, "MH_%.3e_%.2f_%.3e_%.2f_HaloPartCut%d", MH_low, fesc_low, MH_high, fesc_high, HaloPartCut);
      break;

    case 3:
      snprintf(tag, MAX_STRING_LEN - 1, "ejected_%.2f_%.2f_HaloPartCut%d", alpha, beta, HaloPartCut); 
      break;

    /*
    case 4:
      *fesc_local = quasar_baseline * (1 - QuasarFractionalPhoton[p])  + quasar_boosted * QuasarFractionalPhoton[p];
      break;
    */
    case 5:
    case 6:
      snprintf(tag, MAX_STRING_LEN - 1, "AnneMH_%.3e_%.2f_%.3e_%.2f_HaloPartCut%d", MH_low, fesc_low, MH_high, fesc_high, HaloPartCut);      
      break;

    default:
      fprintf(stderr, "The selected fescPrescription is not handled by the switch case in `save_selfcon_grid` in `selfcon_grid.c`.\nPlease add it there.\n");
      return EXIT_FAILURE;

  }

  snprintf(fname_HI, MAX_STRING_LEN, "%s/%s_%s_nionHI_%03d", GridOutputDir, FileNameGalaxies, tag, HighSnap); 

  file_HI = fopen(fname_HI, "wb");
  if (file_HI == NULL)
  {
    fprintf(stderr, "Could not open file %s.\n", fname_HI);
    return EXIT_FAILURE;
  }

  nwritten = myfwrite(grid_towrite->Nion_HI, sizeof(*(grid_towrite->Nion_HI)) * grid_towrite->NumCellsTotal, 1, file_HI);
  if (nwritten != 1)
  {
    fprintf(stderr, "Could not write 1 element of size %zu to file %s, wrote %d instead.\n", sizeof(*(grid_towrite->Nion_HI)) * grid_towrite->NumCellsTotal, fname_HI, nwritten); 
    return EXIT_FAILURE;
  }
  fclose(file_HI);

  return EXIT_SUCCESS;

}

