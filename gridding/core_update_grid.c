#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include <assert.h>
#ifdef MPI
#include <mpi.h>
#endif

#include "core_allvars_grid.h"
#include "core_proto_grid.h"

int32_t update_grid_properties(int32_t filenr)
{

  int32_t snapshot_idx, grid_num_idx, status; 
  int64_t grid_position, gal_idx, good_gals, bad_gals;
  float fesc_local, Ngamma_HI, Ngamma_HeI, Ngamma_HeII;  

  // The outer loop needs to be over snapshots to properly work with the Quasar tracking.
  for (grid_num_idx = 0; grid_num_idx < Grid->NumGrids; ++grid_num_idx) 
  {
    good_gals = 0;
    bad_gals = 0;

    snapshot_idx = ListOutputGrid[grid_num_idx];     

    for (gal_idx = 0; gal_idx < NtotGals; ++gal_idx)
    {
      grid_position = GalGrid[gal_idx].History[snapshot_idx];
      if (grid_position == -1)
      {
        ++bad_gals;
        continue;
      }

      if (grid_position < 0 || grid_position > Grid->NumCellsTotal)
      {
        fprintf(stderr, "The grid index must be between 0 and the GridSize cubed %d cubed = %ld.  The grid index for Galaxy %ld is %ld.\n", GridSize, (long)Grid->NumCellsTotal, (long)gal_idx, (long)grid_position);
        return EXIT_FAILURE;
      }

      if ((GalGrid[gal_idx].StellarMass[snapshot_idx] > 0.0) & (GalGrid[gal_idx].SFR[snapshot_idx] >= 0.0) & (GalGrid[gal_idx].FoFMass[snapshot_idx] > 0.0) & (GalGrid[gal_idx].LenHistory[snapshot_idx] > HaloPartCut)) // Apply some requirements for the galaxy to be included.
      {
        ++good_gals;

        if (fescPrescription == 4)
        {
          update_quasar_tracking(gal_idx, snapshot_idx); 
        }
 
        if (PhotonPrescription == 1)
        {
          calculate_photons(GalGrid[gal_idx].SFR[snapshot_idx], GalGrid[gal_idx].Z[snapshot_idx], &Ngamma_HI, &Ngamma_HeI, &Ngamma_HeII); // Base number of ionizing photons.  Units are log10(Photons/s).
          status = calculate_fesc(gal_idx, snapshot_idx, filenr, &fesc_local);
          if (status == EXIT_FAILURE)
          {
            return EXIT_FAILURE;
          } 

          if (fabs(fesc_local - GalGrid[gal_idx].Gridfesc[snapshot_idx]) > 0.05) 
          {
            printf("Grid %.4e\tSAGE %.4e\tSnap %d\tGal %ld\tTree %d\tFoFNr %d\tGridPos %ld\n",  fesc_local, GalGrid[gal_idx].Gridfesc[snapshot_idx], snapshot_idx, (long)gal_idx, GalGrid[gal_idx].TreeNr, GalGrid[gal_idx].FoFNr[snapshot_idx], (long)grid_position);
          }

#ifdef DEBUG_PYTHON_C
          int32_t count = 0;
          if (count < 10)
          {
            printf("SFR %.10e\tStellar Mass %.10e\tPhotons %.10e\tfesc %.10e\n", GalGrid[gal_idx].SFR[snapshot_idx], log10(GalGrid[gal_idx].StellarMass[snapshot_idx] * 1.0e10 / Hubble_h), Ngamma_HI, fesc_local);
            ++count;
          }
#endif
          sum_photons += pow(10, Ngamma_HI - 50.0);

        }

        if (Ngamma_HI > 0.0)
        {
          Grid->GridProperties[grid_num_idx].Nion_HI[grid_position] += pow(10, Ngamma_HI - 50.0)*fesc_local; // We keep these in units of 10^50 photons/s.
          if (self_consistent == 0)
          {
            Grid->GridProperties[grid_num_idx].Nion_HeI[grid_position] += pow(10, Ngamma_HeI - 50.0)*fesc_local;
            Grid->GridProperties[grid_num_idx].Nion_HeII[grid_position] += pow(10, Ngamma_HeII - 50.0)*fesc_local;
          }
        }

        if (pow(10, Ngamma_HI - 50.0) * fesc_local < 0.0)
        {
          fprintf(stderr, "For galaxy %ld, the number of HI ionizing photons is %.4fe50\n", (long)gal_idx, pow(10, Ngamma_HI - 50.0) * fesc_local); 
          return EXIT_FAILURE;
        }
        if (Grid->GridProperties[grid_num_idx].Nion_HI[grid_position] < 0.0 || Grid->GridProperties[grid_num_idx].Nion_HI[grid_position] > 1e100)
        {
          fprintf(stderr, "For galaxy %ld, cell %ld now has an error number of photons. This number is %.4f e50 photons/s\n", (long)gal_idx, (long)grid_position, Grid->GridProperties[grid_num_idx].Nion_HI[grid_position]);
          return EXIT_FAILURE;
        }

        ++Grid->GridProperties[grid_num_idx].GalCount[grid_position];

        if (self_consistent == 0)
        {      
          Grid->GridProperties[grid_num_idx].SFR[grid_position] += GalGrid[gal_idx].SFR[snapshot_idx];
          Grid->GridProperties[grid_num_idx].StellarMass[grid_position] += GalGrid[gal_idx].StellarMass[snapshot_idx]; 
        } 

        /* These are properties for each galaxy but kept in the grid-struct cause I'm lazy. */
        Grid->GridProperties[grid_num_idx].SnapshotGalaxy[gal_idx] = snapshot_idx;          
        Grid->GridProperties[grid_num_idx].fescGalaxy[gal_idx] = fesc_local;
        Grid->GridProperties[grid_num_idx].MvirGalaxy[gal_idx] = GalGrid[gal_idx].FoFMass[snapshot_idx];
        Grid->GridProperties[grid_num_idx].MstarGalaxy[gal_idx] = GalGrid[gal_idx].StellarMass[snapshot_idx];
        if (Ngamma_HI > 0.0)
        {
          Grid->GridProperties[grid_num_idx].NgammaGalaxy[gal_idx] = pow(10, Ngamma_HI - 50.0);
          Grid->GridProperties[grid_num_idx].NgammafescGalaxy[gal_idx] = pow(10, Ngamma_HI - 50.0)*fesc_local; 
        }
        
      }
      else
      {
        ++bad_gals;
      }
    } // Galaxy loop.

  } // Snapshot loop.
  
  return EXIT_SUCCESS;
}

void count_grid_properties(struct GRID_STRUCT *count_grid) // Count number of galaxies/halos in the grid.
{

  int32_t snapshot_idx, grid_num_idx;

  printf("counting!\n");
  for (grid_num_idx = 0; grid_num_idx < Grid->NumGrids; ++grid_num_idx)
  {

    snapshot_idx = ListOutputGrid[grid_num_idx];     
    
    int64_t GlobalGalCount = 0, SourcesCount = 0, cell_idx;
    float totPhotons_HI = 0, totPhotons_HeI = 0, totPhotons_HeII = 0;

    for (cell_idx = 0; cell_idx < count_grid->NumCellsTotal; ++cell_idx)
    {  
      totPhotons_HI += count_grid->GridProperties[grid_num_idx].Nion_HI[cell_idx];
      GlobalGalCount += count_grid->GridProperties[grid_num_idx].GalCount[cell_idx];

      if (self_consistent == 0)
      {
        totPhotons_HeI += count_grid->GridProperties[grid_num_idx].Nion_HeI[cell_idx];
        totPhotons_HeII += count_grid->GridProperties[grid_num_idx].Nion_HeII[cell_idx];
      }

      if (count_grid->GridProperties[grid_num_idx].Nion_HI[cell_idx] > 0.0)
      {
        ++SourcesCount; 
      }
    }

    printf("At redshift %.3f (Snapshot %d) there was %ld galaxies and [%.4e, %.4e, %.4e]e50 {HI, HeI, HeII}  ionizing Photons emitted per second ([%.4e %.4e %.4e]e50 s^-1 Mpc^-3), spread across %ld cells (%.4f of the total cells).\n", ZZ[snapshot_idx], snapshot_idx, GlobalGalCount, totPhotons_HI, totPhotons_HeI, totPhotons_HeII, totPhotons_HI / pow(BoxSize/Hubble_h,3), totPhotons_HeI / pow(BoxSize/Hubble_h, 3), totPhotons_HeII / pow(BoxSize/Hubble_h,3), (long)SourcesCount, (double)SourcesCount / (double)count_grid->NumCellsTotal); 

  } // Snapshot loop.

}

// INPUT:
// Star formation rate of the galaxy (in units of Msun/yr) (SFR).
// Metallicity (NOT Solar Units) (Z).
//
// OUTPUT/USE:
// Returns the number of HI ionizing photons for the galaxy.  
//
// NOTE: These relationships have been fit manually from STARBURST99 results.  
// DOUBLE NOTE: These relationships assume a constant starformation scenario; a Starburst scenario is completely different.
void calculate_photons(float SFR, float Z, float *Ngamma_HI, float *Ngamma_HeI, float *Ngamma_HeII)
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
}

int32_t calculate_fesc(int p, int i, int filenr, float *fesc_local)
{

  float halomass, ejectedfraction, Mh;

  halomass = GalGrid[p].FoFMass[i];
  ejectedfraction = GalGrid[p].EjectedFraction[i];
  
  if (fescPrescription == 0) 
  {
    *fesc_local = fesc;
  }
  else if (fescPrescription == 1)
  {
    *fesc_local = pow(10,1.0 - 0.2*log10(halomass * 1.0e10 / Hubble_h)); // Deprecated.
  }
  else if (fescPrescription == 2)
  {
    *fesc_local = alpha * pow((halomass * 1.0e10 / Hubble_h), beta); 
  }
  else if (fescPrescription == 3)	
  {
    *fesc_local = alpha * ejectedfraction + beta; 
  }
  else if (fescPrescription == 4)
  {
    *fesc_local = quasar_baseline * (1 - QuasarFractionalPhoton[p])  + quasar_boosted * QuasarFractionalPhoton[p]; 
  }
  else if (fescPrescription == 5)
  {
    Mh = halomass * 1.0e10 / Hubble_h;
    *fesc_local = pow(fesc_low * (fesc_low/fesc_high),(-log10(Mh/MH_low)/log10(MH_high/MH_low)));
    if (*fesc_local > fesc_low)
    {
      *fesc_local = fesc_low;
    }

  } 
  else if (fescPrescription == 6)
  {
    Mh = halomass * 1.0e10 / Hubble_h;
    *fesc_local = 1. - pow((1.-fesc_low) * ((1.-fesc_low)/(1.-fesc_high)),(-log10(Mh/MH_low)/log10(MH_high/MH_low)));
    if (*fesc_local < fesc_low)
    {
      *fesc_local = fesc_low;
    }

  } 
  else if (fescPrescription == 7)
  {
    *fesc_local = alpha * pow(ejectedfraction, beta); 
  }
	
  if (*fesc_local > 1.0)
  {
    fprintf(stderr, "Had fesc_local = %.4f for galaxy %d in file %d with halo mass %.4e (log Msun), Stellar Mass %.4e (log Msun), SFR %.4e (log Msun yr^-1) and Ejected Fraction %.4e\n", *fesc_local, p, filenr, log10(halomass * 1.0e10 / Hubble_h), log10(halomass * 1.0e10 / Hubble_h), log10(GalGrid[p].SFR[i]), ejectedfraction);
    return EXIT_FAILURE; 
  }
   
  if (*fesc_local < 0.0)
  {
    fprintf(stderr, "Had fesc_local = %.4f for galaxy %d in file %d with halo mass %.4e (log Msun), Stellar Mass %.4e (log Msun), SFR %.4e (log Msun yr^-1) and Ejected Fraction %.4e\n", *fesc_local, p, filenr, log10(GalGrid[p].FoFMass[i] * 1.0e10 / Hubble_h), log10(GalGrid[p].StellarMass[i] * 1.0e10 / Hubble_h), log10(GalGrid[p].SFR[i]), GalGrid[p].EjectedFraction[i]);
    return EXIT_FAILURE; 
  }

  return EXIT_SUCCESS;

}

double get_metallicity(double gas, double metals)
{
  double metallicity;

  if(gas > 0.0 && metals > 0.0)
  {
    metallicity = metals / gas;
    if(metallicity < 1.0)
      return metallicity;
    else
      return 1.0;
  }
  else
    return 0.0;

}

int32_t update_quasar_tracking(int64_t gal_idx, int32_t snapshot_idx)
{

  float dt, substep_weight, time_into_snapshot, fraction_into_snapshot; 

 if (GalGrid[gal_idx].QuasarActivity[snapshot_idx] == 1) // A quasar has gone off during this snapshot, time to update properties.
  {
    if (GalGrid[gal_idx].LenMergerGal[snapshot_idx] > HaloPartCut)
    {
      ++QuasarActivityToggle[gal_idx]; // Note, we plus one because we want to be able to handle the case of a quasar going off when the galaxy still is being boosted. 
      QuasarSnapshot[gal_idx] = snapshot_idx;
      TargetQuasarTime[gal_idx] = GalGrid[gal_idx].DynamicalTime[snapshot_idx] * N_dyntime; // How long the quasar will be boosted for.   
      QuasarActivitySubstep[gal_idx] = GalGrid[gal_idx].QuasarSubstep[snapshot_idx]; // What substep did the quasar go off?
      QuasarBoostActiveTime[gal_idx] = 0.0; // How long the quasar boosting has been active for.

      ++QuasarEventsAbovePartCut;
    }
    else
    {
      ++QuasarEventsBelowPartCut;
    }
  }

  if (QuasarActivityToggle[gal_idx] > 0) // This galaxy is having its escape fraction boosted, check to see if we need to turn it off.
  {

    dt = (Age[snapshot_idx - 1] - Age[snapshot_idx]) * UnitTime_in_Megayears; // Time spanned by previous snapshot.
    if (QuasarSnapshot[gal_idx] == snapshot_idx && QuasarActivityToggle[gal_idx] == 1) // If this boosting is due to a quasar going off during this snapshot and the galaxy is NOT under the influence from a previous quasar event then we need to weight the fraction of time the photons are boosted by the substep the quasar went off in. 
    {
      substep_weight = (STEPS - QuasarActivitySubstep[gal_idx]) / STEPS;
    }
    else
    {
      substep_weight = 1.0;
    } 

    QuasarBoostActiveTime[gal_idx] += dt * substep_weight;
    QuasarFractionalPhoton[gal_idx] = substep_weight; // If the quasar turned on part-way through the snapshot, we boost the photons for the remaining time during the snapshot.

    if (QuasarBoostActiveTime[gal_idx] >= TargetQuasarTime[gal_idx]) // The boosted escape fraction needs to be turned off.
    {
      time_into_snapshot = TargetQuasarTime[gal_idx] - (QuasarBoostActiveTime[gal_idx] - dt); // How much extra time into the snapshot does the quasar need to go to reach its target?
      fraction_into_snapshot = time_into_snapshot / dt; // Then what fraction of the snapshot time will this be?
      QuasarFractionalPhoton[gal_idx] = fraction_into_snapshot; 

      //fprintf(stderr, "TargetQuasarTime = %.4f \tQuasarBoostActiveTime = %.4f\tdt = %.4f\ttime_into_snapshot = %.4f\tQuasarFractionalPhoton = %.4f\n", TargetQuasarTime[gal_idx], QuasarBoostActiveTime[gal_idx], dt, time_into_snapshot, QuasarFractionalPhoton[gal_idx]);

      // Reset toggles and trackers. //
      --QuasarActivityToggle[gal_idx];
      QuasarSnapshot[gal_idx] = -1;
      TargetQuasarTime[gal_idx] = 0.0;
      QuasarBoostActiveTime[gal_idx] = 0.0;
      QuasarActivitySubstep[gal_idx] = -1;
    }

  }
  else 
  // One edge case we need to consider is in regards to quasars that were turned off part way through the previous snapshot. // 
  // In this case, we need to fully reset QuasarFractionalPhoton otherwise it will continue to be boosted for a fractional amount. //
  { 
    QuasarFractionalPhoton[gal_idx] = 0.0;
  }
 
  if (QuasarFractionalPhoton[gal_idx] > 1.0)
  {
    fprintf(stderr, "gal_idx = %ld\tQuasarFractionalPhoton[gal_idx] = %.4f\n", (long)gal_idx, QuasarFractionalPhoton[gal_idx]);
    return EXIT_FAILURE;
  }
 
  return EXIT_SUCCESS;

}

#ifdef MPI
struct GRID_STRUCT *MPI_sum_grids()
{

  int32_t status, grid_num_idx;
  struct GRID_STRUCT *master_grid;
  
  master_grid = malloc(sizeof(struct GRID_STRUCT));

  if (ThisTask == 0)
  {
    printf("Trying to initialize the master grid\n");
    status = init_grid(master_grid);
    if (status == EXIT_FAILURE)
    { 
      return NULL;
    }
  }

  for (grid_num_idx = 0; grid_num_idx < Grid->NumGrids; ++grid_num_idx)
  {
    if (ThisTask == 0)
    {
      printf("Reducing grid %d\n", grid_num_idx);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  
    XASSERT(Grid->NumCellsTotal == GridSize*GridSize*GridSize, "Rank %d has %ld grid cells\n", ThisTask, (long)Grid->NumCellsTotal);

    MPI_Reduce(Grid->GridProperties[grid_num_idx].Nion_HI, master_grid->GridProperties[grid_num_idx].Nion_HI, Grid->NumCellsTotal, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); 

    if (self_consistent == 0)
    {
      MPI_Reduce(Grid->GridProperties[grid_num_idx].SFR, master_grid->GridProperties[grid_num_idx].SFR, Grid->NumCellsTotal, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); 
      MPI_Reduce(Grid->GridProperties[grid_num_idx].Nion_HeI, master_grid->GridProperties[grid_num_idx].Nion_HeI, Grid->NumCellsTotal, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(Grid->GridProperties[grid_num_idx].Nion_HeII, master_grid->GridProperties[grid_num_idx].Nion_HeII, Grid->NumCellsTotal, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); 
    MPI_Reduce(Grid->GridProperties[grid_num_idx].GalCount, master_grid->GridProperties[grid_num_idx].GalCount, Grid->NumCellsTotal, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD); 
    }
   
  }

  if (ThisTask == 0)  
    return master_grid;
  else
    return NULL;
}
#endif



