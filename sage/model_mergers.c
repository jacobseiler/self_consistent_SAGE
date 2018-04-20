#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>

#include "core_allvars.h"
#include "core_proto.h"

// Local Proto-Types //

// For BHmodel == 1 need the Eddington Factor. 
const double epsilon = 1.0;
const double t_eddington = 6.6524e-29*2.9979e8/(4*3.1415*6.6741e-11*1.6726e-27) / 3.154e13; // Eddington Timescale (Myr).
#define eddington_factor epsilon/(RadioModeEfficiency*t_eddington)

double estimate_merging_time(int sat_halo, int mother_halo, int ngal)
{
  double coulomb, mergtime, SatelliteMass, SatelliteRadius;

  if(sat_halo == mother_halo) 
  {
    printf("\t\tSnapNum, Type, IDs, sat radius:\t%i\t%i\t%i\t%i\t--- sat/cent have the same ID\n", 
      Gal[ngal].SnapNum, Gal[ngal].Type, sat_halo, mother_halo);
    return -1.0;
  }
  
  coulomb = log(Halo[mother_halo].Len / ((double) Halo[sat_halo].Len) + 1);

  SatelliteMass = get_virial_mass(sat_halo) + Gal[ngal].StellarMass + Gal[ngal].ColdGas;
  SatelliteRadius = get_virial_radius(mother_halo);

  if(SatelliteMass > 0.0 && coulomb > 0.0)
    mergtime = 2.0 *
    1.17 * SatelliteRadius * SatelliteRadius * get_virial_velocity(mother_halo) / (coulomb * G * SatelliteMass);
  else
    mergtime = -1.0;
  
  return mergtime;

}



void deal_with_galaxy_merger(int p, int merger_centralgal, int centralgal, double time, double dt, int halonr, int step, int tree, int ngal)
{
  double mi, ma, mass_ratio;

  XASSERT(Gal[merger_centralgal].IsMerged != 1, "We are trying to merge a galaxy into another galaxy that has already merged.\np = %d \t merger_centralgal = %d \t Gal[merger_centralgal].Type = %d \t Gal[merger_centralgal].mergeType = %d \t Gal[merger_centralgal].IsMerged = %d \t GalaxyNr = %d \t halonr = %d\n", p, merger_centralgal, Gal[merger_centralgal].Type, Gal[merger_centralgal].mergeType, Gal[merger_centralgal].IsMerged, Gal[merger_centralgal].GalaxyNr, halonr);

  // calculate mass ratio of merging galaxies 
  if(Gal[p].StellarMass + Gal[p].ColdGas <
    Gal[merger_centralgal].StellarMass + Gal[merger_centralgal].ColdGas)
  {
    mi = Gal[p].StellarMass + Gal[p].ColdGas;
    ma = Gal[merger_centralgal].StellarMass + Gal[merger_centralgal].ColdGas;
  }
  else
  {
    mi = Gal[merger_centralgal].StellarMass + Gal[merger_centralgal].ColdGas;
    ma = Gal[p].StellarMass + Gal[p].ColdGas;
  }

  Gal[merger_centralgal].LenMergerGal[Halo[Gal[merger_centralgal].HaloNr].SnapNum] = Gal[p].Len;

  if(ma > 0)
    mass_ratio = mi / ma;
  else
    mass_ratio = 1.0;
  add_galaxies_together(merger_centralgal, p);

  // starburst recipe similar to Somerville et al. 2001
  collisional_starburst_recipe(mass_ratio, merger_centralgal, centralgal, time, dt, halonr, 0, step, tree, ngal);

  // grow black hole through accretion from cold disk during mergers, a la Kauffmann & Haehnelt (2000) 
  if(AGNrecipeOn > 0)
  {
    grow_black_hole(merger_centralgal, mass_ratio, step);
  }

  if(mass_ratio > 0.1)
		Gal[merger_centralgal].TimeOfLastMinorMerger = time;

  if(mass_ratio > ThreshMajorMerger)
  {
    make_bulge_from_burst(merger_centralgal);
    Gal[merger_centralgal].TimeOfLastMajorMerger = time;
    Gal[p].mergeType = 2;  // mark as major merger
  }
  else
  {
    Gal[p].mergeType = 1;  // mark as minor merger
  }

  Gal[p].CentralGal = merger_centralgal; 
  
 
}

void grow_black_hole(int merger_centralgal, double mass_ratio, int32_t step)
{
  double metallicity, BHaccrete;

  if(Gal[merger_centralgal].ColdGas > 0.0)
  {
    BHaccrete = BlackHoleGrowthRate * mass_ratio / 
      (1.0 + pow(280.0 / Gal[merger_centralgal].Vvir, 2.0)) * Gal[merger_centralgal].ColdGas;

    // cannot accrete more gas than is available! 
    if(BHaccrete > Gal[merger_centralgal].ColdGas)
      BHaccrete = Gal[merger_centralgal].ColdGas;

    // There are two models possible here for the growth of black holes.

    // The first is the original Croton 2016 model (BHmodel == 0).
    // In this model, the BH accretes all of the mass instantaneousl

    // In ths second (BHmodel == 1), the BH accretes an amount of gas limited by the eddington mass.
    // This will be the amount of mass accreted within the timestep so we smoothly add this over the substeps.

    if (BHmodel == 0)
    {
      metallicity = get_metallicity(Gal[merger_centralgal].ColdGas, Gal[merger_centralgal].MetalsColdGas);

      Gal[merger_centralgal].BlackHoleMass += BHaccrete;
      Gal[merger_centralgal].ColdGas -= BHaccrete;
      Gal[merger_centralgal].MetalsColdGas -= metallicity * (BHaccrete);

      Gal[merger_centralgal].QuasarModeBHaccretionMass += BHaccrete; 
      quasar_mode_wind(merger_centralgal, BHaccrete, step);

      //printf("BHaccrete = %.4e\n", BHaccrete);
    }
    else if (BHmodel == 1)
    {
      Gal[merger_centralgal].BHmass_still_to_accrete += BHaccrete;
      determine_BH_accretion(merger_centralgal);
    }
    else
    {
      fprintf(stderr, "An invalid Black Hole growth model was chosen.\n");
      ABORT(EXIT_FAILURE);
    }

  }
}

void determine_BH_accretion(int32_t p)
{
  // The amount of gas that needs to be accreted onto the black hole is given by Gal[p].BHmass_still_to_accrete.
  // What we want to find here is how much BH mass can be accreted during this time step.
  // Then we will add a fraction of that mass during each substep.

  // First check that there is mass to accrete. 
  if (Gal[p].BHmass_still_to_accrete <= 0.0)
  {
    Gal[p].BHmass_accrete_thisstep = 0.0;
    return;
  }

  double m_eddington;
  double dt = (Age[Gal[p].SnapNum] - Age[Halo[Gal[p].HaloNr].SnapNum]) * UnitTime_in_Megayears;
  //printf("eddington_factor %.4e dt %.4e\n", eddington_factor, dt);
  // Eddington mass will be the MAXIMUM amount of gas that can be accreted within a time step.
  // It's entirely possible that the merger could happen in the middle of a timestep (e.g., at step = 5).
  // In this case, the Eddington mass SHOULD be scaled to be half of the time step.
  // However because we accrete the mass smoothly over the time step regardless (i.e., add 1/10th of the mass during every substep), this doesn't matter.
  // This is because this calculation is simply determining the MAXIMUM amount of gas that we can accrete.  If we're already part way through a timestep we won't accrete that much.
  m_eddington = Gal[p].BlackHoleMass * (exp(eddington_factor * dt) - 1.0);

  //printf("m_eddington %.4e\tBHmass_still_to_accrete %.4e\n", m_eddington, Gal[p].BHmass_still_to_accrete);

  if (Gal[p].BHmass_still_to_accrete > m_eddington && m_eddington > 0.0)
  {
    Gal[p].BHmass_accrete_thisstep = m_eddington; 
  }
  else
  {
    Gal[p].BHmass_accrete_thisstep = Gal[p].BHmass_still_to_accrete; 
  }

}

void smooth_BH_accretion(int32_t p, int32_t step)
{

  double metallicity = get_metallicity(Gal[p].ColdGas, Gal[p].MetalsColdGas);
  double accreted_mass = Gal[p].BHmass_accrete_thisstep / STEPS;

  // First add the mass that's accreted during this timestep to the BH.

  //printf("accreted_mass %.4e\n", accreted_mass);

  Gal[p].BlackHoleMass += accreted_mass; 
  Gal[p].ColdGas -= accreted_mass; 
  Gal[p].MetalsColdGas -=  metallicity * accreted_mass; 
  Gal[p].QuasarModeBHaccretionMass += accreted_mass; 

  // Since we have had a rapid accrete onto the BH, we get an associated pulse of energy.
  quasar_mode_wind(p, accreted_mass, step);

}

void quasar_mode_wind(int gal, float BHaccrete, int32_t step)
{
  float quasar_energy, cold_gas_energy, hot_gas_energy; 

  // work out total energies in quasar wind (eta*m*c^2), cold and hot gas (1/2*m*Vvir^2)
  
  quasar_energy = QuasarModeEfficiency * 0.1 * BHaccrete * (C / UnitVelocity_in_cm_per_s) * (C / UnitVelocity_in_cm_per_s);
  cold_gas_energy = 0.5 * Gal[gal].ColdGas * Gal[gal].Vvir * Gal[gal].Vvir;
  hot_gas_energy = 0.5 * Gal[gal].HotGas * Gal[gal].Vvir * Gal[gal].Vvir;

  if (BHmodel == 1)
  {
    float remaining_energy, hot_energy_ratio, cold_energy_ratio;

    if (cold_gas_energy <= 0.0)
    {
      cold_energy_ratio = quasar_energy / cold_gas_energy;
    }
    else
    {
      cold_energy_ratio = 0.0;
    }

    if (cold_energy_ratio > 1.0)
    {
      Gal[gal].EjectedMass += Gal[gal].ColdGas;
      Gal[gal].MetalsEjectedMass += Gal[gal].MetalsColdGas;

      Gal[gal].ColdGas = 0.0;
      Gal[gal].MetalsColdGas = 0.0;

      remaining_energy = quasar_energy - cold_gas_energy;

      if (hot_gas_energy > 0.0)
      {
        hot_energy_ratio = remaining_energy / hot_gas_energy;
      }
      else
      {
        hot_energy_ratio = 0.0;
      } 
      if (hot_energy_ratio > 1.0)
      {
        Gal[gal].EjectedMass += Gal[gal].HotGas;
        Gal[gal].MetalsEjectedMass += Gal[gal].MetalsHotGas; 

        Gal[gal].HotGas = 0.0;
        Gal[gal].MetalsHotGas = 0.0;

        Gal[gal].QuasarActivity[Halo[Gal[gal].HaloNr].SnapNum] = 1; // Record that there was enough energy to eject all of the cold gas and 50% of the hot gas. 
        Gal[gal].QuasarSubstep[Halo[Gal[gal].HaloNr].SnapNum] = step; // Record at which substep the activity happened. 

      }
      else
      {
        Gal[gal].EjectedMass += Gal[gal].HotGas * hot_energy_ratio;
        Gal[gal].MetalsEjectedMass += Gal[gal].MetalsHotGas * hot_energy_ratio; 

        Gal[gal].HotGas -= Gal[gal].HotGas * hot_energy_ratio;
        Gal[gal].MetalsHotGas -= Gal[gal].HotGas * hot_energy_ratio;
   
      } 

    }
    else
    {
        Gal[gal].EjectedMass += Gal[gal].ColdGas * cold_energy_ratio;
        Gal[gal].MetalsEjectedMass += Gal[gal].MetalsColdGas * cold_energy_ratio;

        Gal[gal].ColdGas -= Gal[gal].ColdGas * cold_energy_ratio;
        Gal[gal].MetalsColdGas -= Gal[gal].MetalsColdGas * cold_energy_ratio;  

    }
  }
  else
  {

    // compare quasar wind and cold gas energies and eject cold
    if(quasar_energy > cold_gas_energy)
    {
      Gal[gal].EjectedMass += Gal[gal].ColdGas;
      Gal[gal].MetalsEjectedMass += Gal[gal].MetalsColdGas;

      Gal[gal].ColdGas = 0.0;
      Gal[gal].MetalsColdGas = 0.0;

    }
   

    // compare quasar wind and cold+hot gas energies and eject hot
    if(quasar_energy > cold_gas_energy + hot_gas_energy)
    {

      Gal[gal].EjectedMass += Gal[gal].HotGas;
      Gal[gal].MetalsEjectedMass += Gal[gal].MetalsHotGas; 

      Gal[gal].HotGas = 0.0;
      Gal[gal].MetalsHotGas = 0.0;

      Gal[gal].QuasarActivity[Halo[Gal[gal].HaloNr].SnapNum] = 1; // Record that there was enough energy to eject cold+hot gas.
      Gal[gal].QuasarSubstep[Halo[Gal[gal].HaloNr].SnapNum] = step; // Record at which substep the activity happened. 

    }
  }

  if(Gal[gal].ColdGas < 0.0) // Some final checks. 
    Gal[gal].ColdGas = 0.0;
  if(Gal[gal].MetalsColdGas < 0.0)
    Gal[gal].MetalsColdGas = 0.0;
  if(Gal[gal].HotGas < 0.0)
    Gal[gal].HotGas = 0.0;
  if(Gal[gal].MetalsHotGas < 0.0)
    Gal[gal].MetalsHotGas = 0.0; 
  
}

void add_galaxies_together(int t, int p)
{
   // t is the index of the central galaxy for the merger.
   // p is the index of the merging galaxy.

  int step, i;

  Gal[t].ColdGas += Gal[p].ColdGas;
  Gal[t].MetalsColdGas += Gal[p].MetalsColdGas;
  
  Gal[t].StellarMass += Gal[p].StellarMass;
  Gal[t].GrandSum += Gal[p].GrandSum;
  Gal[t].MetalsStellarMass += Gal[p].MetalsStellarMass;

  Gal[t].HotGas += Gal[p].HotGas;
  Gal[t].MetalsHotGas += Gal[p].MetalsHotGas;
  
  Gal[t].EjectedMass += Gal[p].EjectedMass;
  Gal[t].MetalsEjectedMass += Gal[p].MetalsEjectedMass;
  
  Gal[t].ICS += Gal[p].ICS;
  Gal[t].MetalsICS += Gal[p].MetalsICS;

  Gal[t].BlackHoleMass += Gal[p].BlackHoleMass;

  // add merger to bulge
	Gal[t].BulgeMass += Gal[p].StellarMass;
	Gal[t].MetalsBulgeMass += Gal[p].MetalsStellarMass;		

  for(step = 0; step < STEPS; step++)
  {
    Gal[t].SfrBulge[step] += Gal[p].SfrDisk[step] + Gal[p].SfrBulge[step];
    Gal[t].SfrBulgeColdGas[step] += Gal[p].SfrDiskColdGas[step] + Gal[p].SfrBulgeColdGas[step];
    Gal[t].SfrBulgeColdGasMetals[step] += Gal[p].SfrDiskColdGasMetals[step] + Gal[p].SfrBulgeColdGasMetals[step];
  }

  XASSERT(t >= 0 && t < GalaxyCounter, "t is out of bounds.  t has a value of %d whereas it should be >= 0 and < GalaxyCounter (%d)\n", t, GalaxyCounter); 
  XASSERT(p >= 0 && p < GalaxyCounter, "p is out of bounds.  p has a value of %d whereas it should be >= 0 and < GalaxyCounter (%d)\n", p, GalaxyCounter); 
  XASSERT(Gal[p].IsMerged == -1, "Gal[p] has already been freed.  p = %d. Gal[p].MergeType = %d\n", p, Gal[p].mergeType);
  XASSERT(Gal[t].IsMerged == -1, "Gal[t] has already been freed.  t = %d. Gal[t].MergeType = %d\n", t, Gal[t].mergeType);

  // Our delayed SN scheme requires the stars formed by current galaxy and also its progenitors; so need to go back through the central galaxy of the merger and add all the stars from the merging galaxy.

  for(i = 0; i < SN_Array_Len; ++i) // Careful that we add up to the current snapshot number (inclusive) as we need to account for the stars just formed.
  {
    Gal[t].Stars[i] += Gal[p].Stars[i];

  }
 
  Gal[t].Total_Stars += Gal[p].Total_Stars;
 
}



void make_bulge_from_burst(int p)
{
  int step;
  
  // generate bulge 
  Gal[p].BulgeMass = Gal[p].StellarMass;
  Gal[p].MetalsBulgeMass = Gal[p].MetalsStellarMass;

  // update the star formation rate 
  for(step = 0; step < STEPS; step++)
  {
    Gal[p].SfrBulge[step] += Gal[p].SfrDisk[step];
    Gal[p].SfrBulgeColdGas[step] += Gal[p].SfrDiskColdGas[step];
    Gal[p].SfrBulgeColdGasMetals[step] += Gal[p].SfrDiskColdGasMetals[step];
    Gal[p].SfrDisk[step] = 0.0;
    Gal[p].SfrDiskColdGas[step] = 0.0;
    Gal[p].SfrDiskColdGasMetals[step] = 0.0;
  }
}

void collisional_starburst_recipe(double mass_ratio, int merger_centralgal, int centralgal, double time, double dt, int halonr, int mode, int step, int tree, int ngal)
{
  double stars, eburst;
  double reheated_mass = 0.0; 
  double mass_metals_new = 0.0; 
  double mass_stars_recycled = 0.0; 
  double ejected_mass = 0.0;
  double NSN = 0.0;

  // This is the major and minor merger starburst recipe of Somerville et al. 2001. 
  // The coefficients in eburst are taken from TJ Cox's PhD thesis and should be more accurate then previous. 

  // the bursting fraction 
  if(mode == 1)
    eburst = mass_ratio;
  else
    eburst = 0.56 * pow(mass_ratio, 0.7); // (1 + ZZ[Gal[merger_centralgal].SnapNum]);

  stars = eburst * Gal[merger_centralgal].ColdGas ;
  if(stars < 0.0)
    stars = 0.0;

  if(SupernovaRecipeOn == 1)
  {    
    if(IRA == 0)
      do_contemporaneous_SN(centralgal, merger_centralgal, dt, &stars, &reheated_mass, &mass_metals_new, &mass_stars_recycled, &ejected_mass, &NSN); 
    else if(IRA == 1)
      do_IRA_SN(centralgal, merger_centralgal, &stars, &reheated_mass, &mass_metals_new, &mass_stars_recycled, &ejected_mass, &NSN); 

    //do_IRA_SN(centralgal, merger_centralgal, &stars, &reheated_mass, &mass_metals_new, &mass_stars_recycled, &ejected_mass); 
   
  } 
  
  if(stars > Gal[merger_centralgal].ColdGas) // we do this check in 'do_current_sn()' but if supernovarecipeon == 0 then we don't do the check.
  {
    double factor = Gal[merger_centralgal].ColdGas / stars; 
    stars *= factor; 
  }

  if (IRA == 0)
    update_from_star_formation(merger_centralgal, stars, dt, step, true, tree, ngal, false); 
  else
    update_from_star_formation(merger_centralgal, stars, dt, step, true, tree, ngal, true);

  //update_from_star_formation(merger_centralgal, stars, dt, step, true, tree, ngal, false); 
 
  update_from_SN_feedback(merger_centralgal, merger_centralgal, reheated_mass, ejected_mass, mass_stars_recycled, mass_metals_new, NSN, dt);

  // check for disk instability
  if(DiskInstabilityOn && mode == 0)
    if(mass_ratio < ThreshMajorMerger)
    check_disk_instability(merger_centralgal, centralgal, halonr, time, dt, step, tree, ngal);

}

void disrupt_satellite_to_ICS(int centralgal, int gal, int tree)
{

  XASSERT(Gal[centralgal].IsMerged == -1, "We are trying to merge a galaxy into another galaxy that has already merged.\nTree = %d \t Gal = %d \t merger_centralgal = %d \t Gal[merger_centralgal].Type = %d \t Gal[merger_centralgal].mergeType = %d \t Gal[merger_centralgal].IsMerged = %d \t GalaxyNr = %d\n", tree, gal, centralgal, Gal[centralgal].Type, Gal[centralgal].mergeType, Gal[centralgal].IsMerged, Gal[centralgal].GalaxyNr);
  
  Gal[centralgal].HotGas += Gal[gal].ColdGas + Gal[gal].HotGas;
  Gal[centralgal].MetalsHotGas += Gal[gal].MetalsColdGas + Gal[gal].MetalsHotGas;
  
  Gal[centralgal].EjectedMass += Gal[gal].EjectedMass;
  Gal[centralgal].MetalsEjectedMass += Gal[gal].MetalsEjectedMass;
  
  Gal[centralgal].ICS += Gal[gal].ICS;
  Gal[centralgal].MetalsICS += Gal[gal].MetalsICS;

  Gal[centralgal].ICS += Gal[gal].StellarMass;
  Gal[centralgal].MetalsICS += Gal[gal].MetalsStellarMass;
  
  // what should we do with the disrupted satellite BH?
  
  Gal[gal].mergeType = 4;  // mark as disruption to the ICS

  Gal[gal].CentralGal = centralgal;  
}

void add_galaxy_to_merger_list(int p)
{
  int32_t j, status;
 
  MergedGal[MergedNr] = Gal[p]; // This is a shallow copy and does not copy the memory the pointers are pointing to.
  status = malloc_temporal_arrays(&MergedGal[MergedNr]);  // Need to malloc arrays for the pointers and then copy over their numeric values.
  if (status == EXIT_FAILURE)
  {
    ABORT(EXIT_FAILURE);
  }

  ++mergedgal_mallocs;

  for (j = 0; j < MAXSNAPS; ++j)
  {
    MergedGal[MergedNr].GridHistory[j] = Gal[p].GridHistory[j];
    MergedGal[MergedNr].GridStellarMass[j] = Gal[p].GridStellarMass[j];
    MergedGal[MergedNr].GridSFR[j] = Gal[p].GridSFR[j];
    MergedGal[MergedNr].GridZ[j] = Gal[p].GridZ[j];
    MergedGal[MergedNr].GridFoFMass[j] = Gal[p].GridFoFMass[j];
    MergedGal[MergedNr].EjectedFraction[j] = Gal[p].EjectedFraction[j]; 
    MergedGal[MergedNr].LenHistory[j] = Gal[p].LenHistory[j];
    MergedGal[MergedNr].GridOutflowRate[j] = Gal[p].GridOutflowRate[j];
    MergedGal[MergedNr].GridInfallRate[j] = Gal[p].GridInfallRate[j];
    MergedGal[MergedNr].GridEjectedMass[j] = Gal[p].GridEjectedMass[j];
    MergedGal[MergedNr].QuasarActivity[j] = Gal[p].QuasarActivity[j];
    MergedGal[MergedNr].DynamicalTime[j] = Gal[p].DynamicalTime[j];
    MergedGal[MergedNr].QuasarSubstep[j] = Gal[p].QuasarSubstep[j];
    MergedGal[MergedNr].GridColdGas[j] = Gal[p].GridColdGas[j];
    MergedGal[MergedNr].LenMergerGal[j] = Gal[p].LenMergerGal[j];
    MergedGal[MergedNr].GridBHMass[j] = Gal[p].GridBHMass[j];
    MergedGal[MergedNr].GridReionMod[j] = Gal[p].GridReionMod[j];
    MergedGal[MergedNr].GridDustColdGas[j] = Gal[p].GridDustColdGas[j];
    MergedGal[MergedNr].GridDustHotGas[j] = Gal[p].GridDustHotGas[j];
    MergedGal[MergedNr].GridDustEjectedMass[j] = Gal[p].GridDustEjectedMass[j];
    MergedGal[MergedNr].GridType[j] = Gal[p].GridType[j];
  }

  for (j = 0; j < SN_Array_Len; ++j)
  {
    MergedGal[MergedNr].Stars[j] = Gal[p].Stars[j];
  }
  ++MergedNr;  
  Gal[p].IsMerged = 1;

  free_temporal_arrays(&Gal[p]); 
  ++gal_frees;
}
 
