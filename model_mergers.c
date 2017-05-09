#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>

#include "core_allvars.h"
#include "core_proto.h"



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



void deal_with_galaxy_merger(int p, int merger_centralgal, int centralgal, double time, double dt, int halonr, int step, int tree)
{
  double mi, ma, mass_ratio;

  XASSERT(Gal[merger_centralgal].IsMerged == -1, "We are trying to merge a galaxy into another galaxy that has already merged.\np = %d \t merger_centralgal = %d \t Gal[merger_centralgal].Type = %d \t Gal[merger_centralgal].mergeType = %d \t Gal[merger_centralgal].IsMerged = %d \t GalaxyNr = %d \t halonr = %d\n", p, merger_centralgal, Gal[merger_centralgal].Type, Gal[merger_centralgal].mergeType, Gal[merger_centralgal].IsMerged, Gal[merger_centralgal].GalaxyNr, halonr);

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

  if(ma > 0)
    mass_ratio = mi / ma;
  else
    mass_ratio = 1.0;
  add_galaxies_together(merger_centralgal, p);

  // grow black hole through accretion from cold disk during mergers, a la Kauffmann & Haehnelt (2000) 
  if(AGNrecipeOn)
  {
    grow_black_hole(merger_centralgal, mass_ratio);
  }
  // starburst recipe similar to Somerville et al. 2001

  collisional_starburst_recipe(mass_ratio, merger_centralgal, centralgal, time, dt, halonr, 0, step, tree);

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



void grow_black_hole(int merger_centralgal, double mass_ratio)
{
  double BHaccrete, metallicity;

  if(Gal[merger_centralgal].ColdGas > 0.0)
  {
    BHaccrete = BlackHoleGrowthRate * mass_ratio / 
      (1.0 + pow(280.0 / Gal[merger_centralgal].Vvir, 2.0)) * Gal[merger_centralgal].ColdGas;

    // cannot accrete more gas than is available! 
    if(BHaccrete > Gal[merger_centralgal].ColdGas)
      BHaccrete = Gal[merger_centralgal].ColdGas;

    metallicity = get_metallicity(Gal[merger_centralgal].ColdGas, Gal[merger_centralgal].MetalsColdGas);
    Gal[merger_centralgal].BlackHoleMass += BHaccrete;
    Gal[merger_centralgal].ColdGas -= BHaccrete;
    Gal[merger_centralgal].MetalsColdGas -= metallicity * BHaccrete;

    Gal[merger_centralgal].QuasarModeBHaccretionMass += BHaccrete;

    quasar_mode_wind(merger_centralgal, BHaccrete);
  }
}



void quasar_mode_wind(int gal, float BHaccrete)
{
  float quasar_energy, cold_gas_energy, hot_gas_energy;

  // work out total energies in quasar wind (eta*m*c^2), cold and hot gas (1/2*m*Vvir^2)
  quasar_energy = QuasarModeEfficiency * 0.1 * BHaccrete * (C / UnitVelocity_in_cm_per_s) * (C / UnitVelocity_in_cm_per_s);
  cold_gas_energy = 0.5 * Gal[gal].ColdGas * Gal[gal].Vvir * Gal[gal].Vvir;
  hot_gas_energy = 0.5 * Gal[gal].HotGas * Gal[gal].Vvir * Gal[gal].Vvir;
   
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
  }
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

  //fprintf(stderr, "Gal[t].mergeType = %d Gal[p].mergeType = %d\n", Gal[t].mergeType, Gal[p].mergeType);

  XASSERT(t >= 0 && t < GalaxyCounter, "t is out of bounds.  t has a value of %d whereas it should be >= 0 and < GalaxyCounter (%d)\n", t, GalaxyCounter); 
  XASSERT(p >= 0 && p < GalaxyCounter, "p is out of bounds.  p has a value of %d whereas it should be >= 0 and < GalaxyCounter (%d)\n", p, GalaxyCounter); 
  XASSERT(Gal[p].IsMerged == -1, "Gal[p] has already been freed.  p = %d. Gal[p].MergeType = %d\n", p, Gal[p].mergeType);
  XASSERT(Gal[t].IsMerged == -1, "Gal[t] has already been freed.  t = %d. Gal[t].MergeType = %d\n", t, Gal[t].mergeType);

  // Our delayed SN scheme requires the stars formed by current galaxy and also its progenitors; so need to go back through the central galaxy of the merger and add all the stars from the merging galaxy.
  //fprintf(stderr, "Adding merger stars\n");
  for(i = 0; i < SN_Array_Len; ++i) // Careful that we add up to the current snapshot number (inclusive) as we need to account for the stars just formed.
  {
  //  fprintf(stderr, "i in add together mergers = %d\n", i); 
    Gal[t].Stars[i] += Gal[p].Stars[i];

  }
  //fprintf(stderr, "Finished adding merger stars\n");   
 
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



void collisional_starburst_recipe(double mass_ratio, int merger_centralgal, int centralgal, double time, double dt, int halonr, int mode, int step, int tree)
{
  double stars, eburst;
  double FracZleaveDiskVal; 
  double reheated_mass = 0.0; 
  double mass_metals_new = 0.0; 
  double mass_stars_recycled = 0.0; 
  double ejected_mass = 0.0;

  // This is the major and minor merger starburst recipe of Somerville et al. 2001. 
  // The coefficients in eburst are taken from TJ Cox's PhD thesis and should be more accurate then previous. 

  // the bursting fraction 
  if(mode == 1)
    eburst = mass_ratio;
  else
    eburst = 0.56 * pow(mass_ratio, 0.7);


  stars = eburst * Gal[merger_centralgal].ColdGas;
  if(stars < 0.0)
    stars = 0.0;
 
  if(SupernovaRecipeOn == 1)
  {    
    if(IRA == 0)
      do_contemporaneous_SN(centralgal, merger_centralgal, dt, &stars, &reheated_mass, &mass_metals_new, &mass_stars_recycled, &ejected_mass); 
    else if(IRA == 1)
      do_IRA_SN(centralgal, merger_centralgal, &stars, &reheated_mass, &mass_metals_new, &mass_stars_recycled, &ejected_mass); 
   
  } 
  
  if(stars > Gal[merger_centralgal].ColdGas) // we do this check in 'do_current_sn()' but if supernovarecipeon == 0 then we don't do the check.
  {
    double factor = Gal[merger_centralgal].ColdGas / stars; 
    stars *= factor; 
  }

  update_from_star_formation(merger_centralgal, stars, dt, step, true, tree); 
  update_from_SN_feedback(merger_centralgal, merger_centralgal, reheated_mass, ejected_mass, mass_stars_recycled, mass_metals_new);

  // check for disk instability
  if(DiskInstabilityOn && mode == 0)
    if(mass_ratio < ThreshMajorMerger)
    check_disk_instability(merger_centralgal, centralgal, halonr, time, dt, step, tree);

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
  int j;

  MergedGal[MergedNr] = Gal[p]; // This is a shallow copy and does not copy the memory the pointers are pointing to.
  malloc_grid_arrays(&MergedGal[MergedNr]);  // Need to malloc arrays for the pointers and then copy over their numeric values.

  for (j = 0; j < MAXSNAPS; ++j)
  {
    MergedGal[MergedNr].GridHistory[j] = Gal[p].GridHistory[j];
    MergedGal[MergedNr].GridStellarMass[j] = Gal[p].GridStellarMass[j];
    MergedGal[MergedNr].GridSFR[j] = Gal[p].GridSFR[j];
    MergedGal[MergedNr].GridZ[j] = Gal[p].GridZ[j];
    MergedGal[MergedNr].GridCentralGalaxyMass[j] = Gal[p].GridCentralGalaxyMass[j];
    MergedGal[MergedNr].MfiltGnedin[j] = Gal[p].MfiltGnedin[j];
    MergedGal[MergedNr].MfiltSobacchi[j] = Gal[p].MfiltSobacchi[j];
    MergedGal[MergedNr].EjectedFraction[j] = Gal[p].EjectedFraction[j]; 
    MergedGal[MergedNr].LenHistory[j] = Gal[p].LenHistory[j];
    MergedGal[MergedNr].GridOutflowRate[j] = Gal[p].GridOutflowRate[j]; 
  }
 
  for (j = 0; j < SN_Array_Len; ++j)
  {
    MergedGal[MergedNr].Stars[j] = Gal[p].Stars[j];
  }

  free_grid_arrays(&Gal[p]);
  Gal[p].IsMerged = 1; 
  ++MergedNr;
}
 
