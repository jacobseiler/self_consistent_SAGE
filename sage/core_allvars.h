#ifndef ALLVARS_H
#define ALLVARS_H

#include <stdio.h>
#include <stdbool.h>
#include <gsl/gsl_rng.h>
#include "core_simulation.h"
#include <stdint.h>

#define ABORT(sigterm)                                                  \
do {                                                                \
  printf("Error in file: %s\tfunc: %s\tline: %i\n", __FILE__, __FUNCTION__, __LINE__); \
  myexit(sigterm);                                                \
} while(0)

#define  STEPS 10       // Number of integration intervals between two snapshots
#define  MAXGALFAC 1
#define  ALLOCPARAMETER 10.0
#define  MAX_NODE_NAME_LEN 50
#define  ABSOLUTEMAXSNAPS 1000

#define  MAXLEN      1024
#define  MAX_STRING_LEN      1024
#define  GRAVITY     6.672e-8
#define  SOLAR_MASS  1.989e33
#define  SOLAR_LUM   3.826e33
#define  RAD_CONST   7.565e-15
#define  AVOGADRO    6.0222e23
#define  BOLTZMANN   1.3806e-16
#define  GAS_CONST   8.31425e7
#define  C           2.9979e10
#define  PLANCK      6.6262e-27
#define  CM_PER_MPC  3.085678e24
#define  PROTONMASS  1.6726e-24
#define  HUBBLE      3.2407789e-18   /* in h/sec */

#define  SEC_PER_MEGAYEAR   3.155e13
#define  SEC_PER_YEAR       3.155e7

#define	 CUBE(x) (x*x*x)

#ifdef NDEBUG
#define XASSERT(EXP, ...)                                do{} while(0)
#else
#define XASSERT(EXP, ...)                                              \
    do { if (!(EXP)) {                                                  \
            printf("Error in file: %s\tfunc: %s\tline: %d with expression `"#EXP"'\n", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                        \
            fflush(stdout);                                             \
            ABORT(EXIT_FAILURE);                                         \
        } \
    } while (0)
#endif

#ifdef NDEBUG
#define XPRINT(EXP, ...)                                do{} while(0)
#else
#define XPRINT(EXP, ...)                                              \
    do { if (!(EXP)) {                                                  \
            printf("Warning in file: %s\tfunc: %s\tline: %d with expression `"#EXP"'\n", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                        \
            fflush(stdout);                                             \
        } \
    } while (0)
#endif

// This structure contains the properties used within the code
struct GALAXY
{
  int   SnapNum;
  int   Type;

  int   GalaxyNr;
  int   CentralGal;
  int   HaloNr;
  int   TreeNr; 
  long long  MostBoundID;

  int   mergeType;  //0=none; 1=minor merger; 2=major merger; 3=disk instability; 4=disrupt to ICS
  int   mergeIntoID;
  int   mergeIntoSnapNum;
  float   dT;
  float Age;
  // (sub)halo properties
  float Pos[3];
  float Vel[3];
  int   Len;   
  float Mvir;
  float deltaMvir;
  float CentralMvir;
  float Rvir;
  float Vvir;
  float Vmax;

  // baryonic reservoirs 
  float ColdGas;
  float StellarMass;
  float BulgeMass;
  float HotGas;
  float EjectedMass;
  float BlackHoleMass;
  float ICS;

  // metals
  float MetalsColdGas;
  float MetalsStellarMass;
  float MetalsBulgeMass;
  float MetalsHotGas;
  float MetalsEjectedMass;
  float MetalsICS;

  // to calculate magnitudes
  float SfrDisk[STEPS];
  float SfrBulge[STEPS];
  float SfrDiskColdGas[STEPS];
  float SfrDiskColdGasMetals[STEPS];
  float SfrBulgeColdGas[STEPS];
  float SfrBulgeColdGasMetals[STEPS];

  // misc 
  float DiskScaleRadius;
  float MergTime;
  double Cooling;
  double Heating;
  float r_heat;
  float QuasarModeBHaccretionMass;
  float TimeOfLastMajorMerger;
  float TimeOfLastMinorMerger;
  float OutflowRate;
	float TotalSatelliteBaryons;

  // infall properties
  float infallMvir;
  float infallVvir;
  float infallVmax;

  int32_t GridPos;
  int *GridHistory;
  float *GridColdGas;
  float *GridHotGas;
  float *GridStellarMass;
  float *GridSFR;
  float *GridZ;
  float *GridFoFMass;
  float *EjectedFraction;
  int *LenHistory;
  double *Stars; 
  float *GridOutflowRate;
  float *GridInfallRate;
  float *GridEjectedMass;
  int *QuasarActivity;
  float *DynamicalTime; 
  int *QuasarSubstep;
  int *LenMergerGal;
  float *GridBHMass;      
  float *GridReionMod;
  float *GridDustColdGas;
  float *GridDustHotGas;
  float *GridDustEjectedMass;
  int32_t *GridType;
 
  double StellarAge_Numerator;
  double StellarAge_Denominator;
  double Total_SF_Time;
  double Total_Stars;

  int IsMerged;
  double GrandSum;
  int IsMalloced;

  // Delayed SN Properties
  float reheated_mass;
  float ejected_mass;
  float mass_stars_recycled;
  float mass_metals_new; 
  float NSN; // Number of supernova within a time step.  Can be fractional. 
 
  // Dust Properties
  float DustColdGas;
  float DustHotGas;
  float DustEjectedMass;

}
*Gal, *HaloGal, *MergedGal;


// auxiliary halo data
struct halo_aux_data   
{
  int DoneFlag;
  int HaloFlag;
  int NGalaxies;
  int FirstGalaxy;
}
*HaloAux;

// For the fully self consistent model we read in a list of reionization modifiers for each halo.
// These lists only contain those halos with a reionization modifier less than 0.9999.
// We will read if a list for each of the previous snapshots below the one we are currently on.
// E.g., if we are applying reionization feedback for snapshot 40, we will read in lists for snapshot 0 to 39. 
// There is one list for each tree file.
struct REIONMOD_STRUCT
{

  int32_t NumLists; // Will be reionization snapshot number.
  struct REIONMOD_LIST
  {
    int32_t NHalos_Found; // Number of halos that have been matched.
    int32_t NHalos_Ionized; // Number of halos in the list.
    int64_t *HaloID; // This is a unique ID for each tree file.  The most significant 32 bits (left-most) is the tree number, and the least significant (right-most) bits is the halonr within the tree.
    float *ReionMod; // Reionization Modifer for each halo.
  }*ReionMod_List; 
}*ReionList;

// Grid for doing photoionization feedback.
// This will contain multiple grids, one for each redshift. 
struct GRID_STRUCT 
{

  int32_t GridSize;   
  uint64_t NumCellsTotal;
  int32_t NumGrids;

  double *ReionRedshift; // This is the redshift the grid cell was ionized at.
  struct PHOTO_GRID 
  {     
    double *PhotoRate; // Photoionization Rate (s^-1).
    int32_t valid_grid; // This will control whether data has been read in for this snapshot. Useful for the early snapshots where we don't have any ionization.
  }*PhotoGrid;
}*Grid;


struct SELFCON_GRID_STRUCT 
{

  int32_t GridSize;   
  int32_t NumCellsTotal;  

  float *Nion_HI;
  int32_t *GalCount;

}*SelfConGrid;

extern int    FirstFile;    // first and last file for processing 
extern int    LastFile;

extern int count_onehalo;

extern int    Ntrees;      // number of trees in current file 
extern int    NumGals;     // Total number of galaxies stored for current tree 
extern int    MaxGals;     // Maximum number of galaxies allowed for current tree  
extern int    FoF_MaxGals;

extern int    GalaxyCounter;     // unique galaxy ID for main progenitor line in tree

extern int    LastSnapShotNr;

extern char   OutputDir[512];
extern char   GridOutputDir[512];
extern char   FileNameGalaxies[512];
extern char   TreeName[512];
extern char   TreeExtension[512];
extern char   SimulationDir[512];
extern char   IonizationDir[512];
extern char   FileWithSnapList[512];
extern char   PhotoionDir[512];
extern char   PhotoionName[512];
extern char   ReionRedshiftName[512];
extern int    ReionSnap;

extern int    TotHalos;
extern int    TotGalaxies[ABSOLUTEMAXSNAPS];
extern int    *TreeNgals[ABSOLUTEMAXSNAPS];

extern int    *FirstHaloInSnap;

extern int    *TreeNHalos;
extern int    *TreeFirstHalo;

#ifdef MPI
extern int ThisTask, NTask, nodeNameLen;
extern char *ThisNode;
#endif

extern int IMF;

extern double Omega;
extern double OmegaLambda;
extern double PartMass;
extern double BoxSize;
extern int GridSize;
extern int self_consistent;
extern double Hubble_h;
extern double EnergySNcode, EnergySN;

// recipe flags 
extern int    ReionizationOn;
extern int    SupernovaRecipeOn;
extern int    DiskInstabilityOn;
extern int    AGNrecipeOn;
extern int    SFprescription;

// recipe parameters 
extern double RecycleFraction;
extern double Yield;
extern double FracZleaveDisk;
extern double ReIncorporationFactor;
extern double ThreshMajorMerger;
extern double BaryonFrac;
extern double SfrEfficiency;
extern double FeedbackReheatingEpsilon;
extern double FeedbackEjectionEfficiency;
extern double RadioModeEfficiency;
extern double QuasarModeEfficiency;
extern double BlackHoleGrowthRate;
extern double Reionization_z0;
extern double Reionization_zr;
extern double ThresholdSatDisruption;

// Parameters for the gridding with self_consistent
extern int fescPrescription;
extern double fesc;

extern double MH_low;
extern double fesc_low;
extern double MH_high;
extern double fesc_high;

extern double alpha;
extern double beta;

extern double quasar_baseline;
extern double quasar_boosted;
extern double N_dyntime;

extern int HaloPartCut;

extern double UnitLength_in_cm,
  UnitTime_in_s,
  UnitVelocity_in_cm_per_s,
  UnitMass_in_g,
  RhoCrit,
  UnitPressure_in_cgs,
  UnitDensity_in_cgs,
  UnitCoolingRate_in_cgs,
  UnitEnergy_in_cgs,
  UnitTime_in_Megayears, 
  G,
  Hubble,
  a0, ar;

extern int    ListOutputSnaps[ABSOLUTEMAXSNAPS];

extern double ZZ[ABSOLUTEMAXSNAPS];
extern double AA[ABSOLUTEMAXSNAPS];
extern double Age[ABSOLUTEMAXSNAPS];

extern int    MAXSNAPS;
extern int    NOUT;
extern int    Snaplistlen;

extern gsl_rng *random_generator;

extern int TreeID;
extern int FileNum;
extern int neg_cell;
extern int pos_cell;

extern double ejectedmass_total;
extern double metalsejectedmass_total;

extern int MergedNr;
extern int TotMerged;
extern int *TreeNMergedgals; 
extern int MaxMergedGals;     // Maximum number of galaxies allowed for current tree  

extern int zeromass_count;
extern int suppression_count;
extern int previous_tree;
extern int lowmass_halo;

extern double smallest_mass;

extern double IMF_norm;
extern double IMF_slope;
extern double Eta_SNII;
extern double m_SNII;
extern int IRA; 
extern int TimeResolutionSN;
extern int SN_Array_Len; 
extern int Time_SFH;

extern double alpha_energy;
extern double V_energy;
extern double beta_energy;
extern double alpha_mass;
extern double V_mass;
extern double beta_mass; 
extern double epsilon_mass_max;

extern int RescaleSN;
extern int mergedgal_mallocs;
extern int gal_mallocs;
extern int ismerged_mallocs;

extern int mergedgal_frees;
extern int gal_frees;
extern int ismerged_frees;

extern int *gal_to_free;

extern int outside_box;
extern int inside_box;

extern int count_Mvir;
extern int count_Len;

extern long count_gal;

extern int32_t LowSnap;
extern int32_t HighSnap;

#endif  // #ifndef ALLVARS_H
