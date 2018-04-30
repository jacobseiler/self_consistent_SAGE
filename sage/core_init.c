#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "core_allvars.h"
#include "core_proto.h"

void init(void)
{
  int i;

  count_gal = 0;
  random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);
  gsl_rng_set(random_generator, 42);	 // start-up seed 

  set_units();
  srand((unsigned) time(NULL));

  read_snap_list();

  for(i = 0; i < MAXSNAPS; i++)
  {
    ZZ[i] = 1 / AA[i] - 1;
    Age[i] = time_to_present(ZZ[i]);

  }

  a0 = 1.0 / (1.0 + Reionization_z0);
  ar = 1.0 / (1.0 + Reionization_zr);

  read_cooling_functions();

  count_onehalo = 0;  

  zeromass_count = 0;
  suppression_count = 0;
  previous_tree = 0;

  smallest_mass = 100000;
  lowmass_halo = 0;

  outside_box = 0;
  inside_box = 0;

  count_Mvir = 0;
  count_Len = 0;
  if (IMF == 1)
  {
    // Salpeter IMF //
    IMF_norm =0.1706;
    IMF_slope = -2.35;
    Eta_SNII = 7.432e-3; //Number fraction of stars that will end their lives as type II supernovae.
    m_SNII = 0.144; // Mass fraction of stars that will end their lives as type II supernovae.

  } else if (IMF == 2)
  {
    // Chabrier IMF //
    IMF_norm = 0.23638; 
    IMF_slope = -2.3;
    Eta_SNII = 1.1893e-2; //Number fraction of stars that will end their lives as type II supernovae.
    m_SNII = 0.23638; // Mass fraction of stars that will end their lives as type II supernovae.

  }
 
  // Tiamat Parameters //

  V_energy = 70.0;
  alpha_energy = 0.5;
  beta_energy = 10.0;

  V_mass = 70.0; 
  alpha_mass = 6.0; 
  beta_mass = 10.0; 

  epsilon_mass_max = 30.0;

  if(TimeResolutionSN > 50)
  {
    fprintf(stderr, "The selected time resolution for SN feedback (TimeResolutionSN) is set too high (%d Myr).  Using TimeResolutionSN > 50Myr is the same as using the instantaneous recycling approximation; set 'IRA' to 1 instead!\n", TimeResolutionSN); 
    ABORT(EXIT_FAILURE);  
  } else if(TimeResolutionSN > 35)
  {
    fprintf(stderr, "Your selected time resolution for SN feedback (TimeResolutionSN) is quite high (%d Myr).  Beyond 50Myr the instantaneous recycling approximation is valid hence with your value it would likely be correct to set 'IRA' to 1.\n", TimeResolutionSN);
  } else
  {
    Time_SFH = 0;
    SN_Array_Len = 0;
    while(Time_SFH < 50)
    {
      Time_SFH += TimeResolutionSN;
      ++SN_Array_Len;
    }

  }
  
  mergedgal_mallocs = 0;
  gal_mallocs = 0 ;

  mergedgal_frees = 0;
  gal_frees = 0;
 
}

void set_units(void)
{

  UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;
  UnitTime_in_Megayears = UnitTime_in_s / SEC_PER_MEGAYEAR;
  G = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);
  UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3);
  UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2);
  UnitCoolingRate_in_cgs = UnitPressure_in_cgs / UnitTime_in_s;
  UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2);

  EnergySNcode = EnergySN / UnitEnergy_in_cgs * Hubble_h;

  // convert some physical input parameters to internal units 
  Hubble = HUBBLE * UnitTime_in_s;

  // compute a few quantitites 
  RhoCrit = 3 * Hubble * Hubble / (8 * M_PI * G);

}

void read_snap_list(void)
{
  FILE *fd;
  char fname[1000];

  sprintf(fname, "%s", FileWithSnapList);

  if(!(fd = fopen(fname, "r")))
  {
    printf("can't read output list in file '%s'\n", fname);
    ABORT(0);
  }

  Snaplistlen = 0;
  do
  {
    if(fscanf(fd, " %lg ", &AA[Snaplistlen]) == 1)
      Snaplistlen++;
    else
      break;
  }
  while(Snaplistlen < MAXSNAPS);

  fclose(fd);

#ifdef MPI
  if(ThisTask == 0)
#endif
    printf("found %d defined times in snaplist\n", Snaplistlen);
}

double time_to_present(double z)
{
#define WORKSIZE 1000
  gsl_function F;
  gsl_integration_workspace *workspace;
  double time, result, abserr;

  workspace = gsl_integration_workspace_alloc(WORKSIZE);
  F.function = &integrand_time_to_present;

  gsl_integration_qag(&F, 1.0 / (z + 1), 1.0, 1.0 / Hubble,
    1.0e-8, WORKSIZE, GSL_INTEG_GAUSS21, workspace, &result, &abserr);

  time = 1 / Hubble * result;

  gsl_integration_workspace_free(workspace);

  // return time to present as a function of redshift 
  return time;
}

double integrand_time_to_present(double a, void *param)
{
  (void)(param); // Avoids triggering unused-parameter warning.
  return 1 / sqrt(Omega / a + (1 - Omega - OmegaLambda) + OmegaLambda * a * a);
}

