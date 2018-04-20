#include "core_allvars.h"


// galaxy data 
struct GALAXY			
  *Gal, *HaloGal, *MergedGal;

struct halo_data *Halo;

struct GRID_STRUCT
  *Grid;

struct REIONMOD_STRUCT
  *ReionList;

struct SELFCON_GRID_STRUCT
  *SelfConGrid;

// auxiliary halo data 
struct halo_aux_data		
  *HaloAux;


// misc 
int FirstFile;
int LastFile;
int MaxGals;
int FoF_MaxGals;
int Ntrees;			   // number of trees in current file 
int NumGals;			 // Total number of galaxies stored for current tree 

int GalaxyCounter; // unique galaxy ID for main progenitor line in tree

char OutputDir[512];
char GridOutputDir[512];
char FileNameGalaxies[512];
char TreeName[512];
char TreeExtension[512];
char SimulationDir[512];
char IonizationDir[512];
char FileWithSnapList[512];
char PhotoionDir[512];
char PhotoionName[512];
char ReionRedshiftName[512];
int  ReionSnap;

int TotHalos;
int TotGalaxies[ABSOLUTEMAXSNAPS];
int *TreeNgals[ABSOLUTEMAXSNAPS];

int LastSnapShotNr;

int count_onehalo;

int *FirstHaloInSnap;
int *TreeNHalos;
int *TreeFirstHalo;

#ifdef MPI
int ThisTask, NTask, nodeNameLen;
char *ThisNode;
#endif

int IMF;

double Omega;
double OmegaLambda;
double Hubble_h;
double PartMass;
double BoxSize;
int GridSize;
int self_consistent;
double EnergySNcode, EnergySN;

// recipe flags 
int ReionizationOn;
int SupernovaRecipeOn;
int DiskInstabilityOn;
int AGNrecipeOn;
int SFprescription;
int BHmodel;


// recipe parameters 
double RecycleFraction;
double Yield;
double FracZleaveDisk;
double ReIncorporationFactor;
double ThreshMajorMerger;
double BaryonFrac;
double SfrEfficiency;
double FeedbackReheatingEpsilon;
double FeedbackEjectionEfficiency;
double RadioModeEfficiency;
double QuasarModeEfficiency;
double BlackHoleGrowthRate;
double Reionization_z0;
double Reionization_zr;
double ThresholdSatDisruption;

// Parameters for the gridding with self_consistent
int fescPrescription;
double fesc;
double MH_low;
double fesc_low;
double MH_high;
double fesc_high;

double alpha;
double beta;

double quasar_baseline;
double quasar_boosted;
double N_dyntime;

int HaloPartCut;

// more misc 
double UnitLength_in_cm,
  UnitTime_in_s,
  UnitVelocity_in_cm_per_s,
  UnitMass_in_g,
  RhoCrit,
  UnitPressure_in_cgs,
  UnitDensity_in_cgs, UnitCoolingRate_in_cgs, UnitEnergy_in_cgs, UnitTime_in_Megayears, G, Hubble, a0, ar;

int ListOutputSnaps[ABSOLUTEMAXSNAPS];

double ZZ[ABSOLUTEMAXSNAPS];
double AA[ABSOLUTEMAXSNAPS];
double Age[ABSOLUTEMAXSNAPS];

int MAXSNAPS;
int NOUT;
int Snaplistlen;

gsl_rng *random_generator;

double ejectedmass_total;
double metalsejectedmass_total;

int TreeID;
int FileNum;

int neg_cell;
int pos_cell;

int MergedNr;
int TotMerged;
int *TreeNMergedgals;
int MaxMergedGals;     // Maximum number of galaxies allowed for current tree  

int zeromass_count;
int suppression_count;
int previous_tree;
int lowmass_halo;

double smallest_mass;

double IMF_norm;
double IMF_slope;
double Eta_SNII;
double m_SNII;
int IRA; 
int TimeResolutionSN;
int SN_Array_Len; 
int Time_SFH;

double alpha_energy;
double V_energy;
double beta_energy;
double alpha_mass;
double V_mass;
double beta_mass; 
double epsilon_mass_max;

int RescaleSN;

int mergedgal_mallocs;
int gal_mallocs;
int ismerged_mallocs;

int mergedgal_frees;
int gal_frees;
int ismerged_frees;

int *gal_to_free;

int outside_box;
int inside_box;

int count_Mvir;
int count_Len;
long count_gal;

int32_t LowSnap;
int32_t HighSnap;
