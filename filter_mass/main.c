#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include <assert.h>

#ifdef MPI
#include <mpi.h>
#endif

#include "read_parameter_file.h"
#include "tree_io.h"
#include "reionization.h"
#include "save.h"

#define MAXLEN 1024
#define	CUBE(x) (x*x*x)
#define ABSOLUTEMAXSNAPS 999

// Local Structs //

// Local Variables //

struct SAGE_PARAMETERS *params;
int32_t ThisSnap, first_run;
#ifdef MPI
int ThisTask, NTask, nodeNameLen;
char *ThisNode;
#endif

// Proto-types //

int32_t parse_params(int32_t argc, char **argv, struct SAGE_PARAMETERS *params);
int32_t read_snap_list(struct SAGE_PARAMETERS *params);
void myexit(int signum);

// Functions //

void bye()
{
#ifdef MPI
  MPI_Finalize();
  free(ThisNode);
#endif
}

int32_t parse_params(int32_t argc, char **argv, struct SAGE_PARAMETERS *params)
{

  int32_t status;

  if (argc != 4)
  {
    fprintf(stderr, "\n\n"); 
    fprintf(stderr, "This code reads in photoionization grids and pre-processes the dark matter halo trees to create a list of filtering masses.\n");
    fprintf(stderr, "./create_filter mass <SAGE Input Parameter File> <Snapshot Number> <First Run Flag>\n");
    fprintf(stderr, "The snapshot number should be the snapshot we're creating the filtering masses for.\n");
    fprintf(stderr, "First run flag is 0 or 1 denoting whether this is the first snapshot we're creating the list for. Since we want one list for each tree file that contains all snapshots, the first time the code is run we need to create the file and the other times we simply need to append.\n");
    fprintf(stderr, "\n\n"); 
    return EXIT_FAILURE; 
  }
  
  status = read_parameter_file(argv[1], params); 
  if (status != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  } 

  ThisSnap = atoi(argv[2]);

  first_run = atoi(argv[3]);
  if (first_run < 0 || first_run > 1)
  {
    fprintf(stderr, "The first run flag can only be 0 or 1.\n");
    return EXIT_FAILURE;
  }

  printf("\n\n");
  printf("==================================================\n");
  printf("Executing with\nGalaxy Name: %s\nTree Directory: %s\nTree Name: %s\nNumber of Tree Files: %d\nPhotoionization Directory: %s\nPhotoionization Name: %s\nReionization Redshift Name: %s\nGridSize: %d\nSnapshot Number: %d\nFirst Run Flag: %d\n", params->FileNameGalaxies, params->TreeDir, params->TreeName, params->LastFile - params->FirstFile + 1, params->PhotoionDir, params->PhotoionName, params->ReionRedshiftName, params->GridSize, ThisSnap, first_run); 
  printf("==================================================\n\n");
  printf("\n\n");
 
  return EXIT_SUCCESS;
}

int32_t read_snap_list(struct SAGE_PARAMETERS *params) 
{

  FILE *fd;
  char fname[1000];
  double AA[ABSOLUTEMAXSNAPS];

  int32_t i, Snaplistlen = 0;

  snprintf(fname, MAXLEN, "%s", params->SnapListFile);

  if(!(fd = fopen(fname, "r")))
  {
    printf("can't read output list in file '%s'\n", fname);
    return EXIT_FAILURE;  
  }

  Snaplistlen = 0;
  do
  {
    if(fscanf(fd, " %lg ", &AA[Snaplistlen]) == 1)
      Snaplistlen++;
    else
      break;
  }
  while(Snaplistlen < (params->LastSnapshotNr + 1));

  fclose(fd);

  for (i = 0; i < Snaplistlen; ++i)
  {  
    params->ZZ[i] = 1 / AA[i] - 1;  
  } 

  return EXIT_SUCCESS;

}

void myexit(int signum)
{
#ifdef MPI
  fprintf(stderr, "Task: %d\tnode: %s\tis exiting\n\n\n", ThisTask, ThisNode);
  MPI_Abort(MPI_COMM_WORLD, signum);
#else
  fprintf(stderr, "We're exiting\n\n\n");
	exit(signum);
#endif

}

int main(int argc, char **argv)
{

  int32_t status, filenr, Ntrees, totNHalos, *TreeNHalos, treenr, NHalos_Ionized, NHalos_In_Regions, NHalos_ThisSnap;
  int64_t *HaloID;
  float *ReionMod, sum_ReionMod; 

  struct HALO_STRUCT *Halos; 
  struct GRID_STRUCT *Grid;

#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  ThisNode = malloc(MPI_MAX_PROCESSOR_NAME * sizeof(char));

  MPI_Get_processor_name(ThisNode, &nodeNameLen);
  if (nodeNameLen >= MPI_MAX_PROCESSOR_NAME)
  {
    printf("Node name string not long enough!...\n");
    ABORT(EXIT_FAILURE);
  }
#endif

  atexit(bye);
  params = malloc(sizeof(struct SAGE_PARAMETERS));
  if (params == NULL)
  {
    fprintf(stderr, "Cannot allocate memory for parameter struct.\n");
    ABORT(EXIT_FAILURE); 
  }
 
  status = parse_params(argc, argv, params); // Set the input parameters.
  if (status != EXIT_SUCCESS)
  {
    ABORT(EXIT_FAILURE);
  }

  status = read_snap_list(params); // Get the simulation redshifts.
  if (status != EXIT_SUCCESS)
  {
    ABORT(EXIT_FAILURE);
  }

  Grid = malloc(sizeof(struct GRID_STRUCT));
  if (Grid == NULL)
  {
    fprintf(stderr, "Could not allocate memory for Grid struct.\n");
    ABORT(EXIT_FAILURE);
  }

  status = read_grid(ThisSnap, first_run, params, Grid); // Read the reionization redshift and photoionization grid. 
  if (status != EXIT_SUCCESS)
  {
    ABORT(EXIT_FAILURE);
  }
   
#ifdef MPI 
  if (NTask > params->LastFile + 1)
  {
    fprintf(stderr, "Attempted to run filter_mass with %d Tasks.  However the last file is %d.\n", NTask, params->LastFile);
    ABORT(EXIT_FAILURE);
  }

  for(filenr = params->FirstFile + ThisTask; filenr < params->LastFile + 1 ; filenr += NTask)
#else
  for(filenr = params->FirstFile; filenr < params->LastFile + 1; filenr++)
#endif
  {
    NHalos_Ionized = 0;
    NHalos_In_Regions = 0;
    NHalos_ThisSnap = 0;
    
    status = load_tree_table(filenr, params, &Ntrees, &totNHalos, &TreeNHalos); // Loads the table for this file.
    if (status != EXIT_SUCCESS)
    {
      ABORT(EXIT_FAILURE);
    }       

#ifdef DEBUG_TREES
    int32_t i;
    int32_t maxhalos = -1, minhalos = 1e5;
    for (i = 0; i < Ntrees; ++i)
    {
      if (TreeNHalos[i] > maxhalos)
        maxhalos = TreeNHalos[i];
    
      if (TreeNHalos[i] < minhalos)
        minhalos = TreeNHalos[i];
    }
    printf("The smallest tree had %d halos and the largest had %d halos\n", minhalos, maxhalos);
#endif

    status = allocate_array_memory(totNHalos, &HaloID, &ReionMod); // Memory for the output arrays. 
    if (status != EXIT_SUCCESS)
    {
      ABORT(EXIT_FAILURE);
    }

    for (treenr = 0; treenr < Ntrees; ++treenr)
    {
      Halos = malloc(sizeof(struct HALO_STRUCT) * TreeNHalos[treenr]); // Allocate the memory in main to keep consistency.
      if (Halos == NULL)
      {
        fprintf(stderr, "Could not allocate memory for Halos in tree %d\n", treenr);
        ABORT(EXIT_FAILURE); 
      }

      status = load_halos(treenr, TreeNHalos[treenr], Halos); // Loads the halos for this tree.
      if (status != EXIT_SUCCESS)
      {
        ABORT(EXIT_FAILURE);
      }



      // Now time to go through all the halos in this tree, determine those at the Snapshot specified and the associate reionization modifier (if it's within an ionized cell).
      status = populate_halo_arrays(filenr, treenr, TreeNHalos[treenr], ThisSnap, first_run, Halos, Grid, params, &HaloID, &ReionMod, &NHalos_ThisSnap, &NHalos_Ionized, &NHalos_In_Regions, &sum_ReionMod);
      if (status != EXIT_SUCCESS)
      {
        ABORT(EXIT_FAILURE);
      }

      free(Halos);
    } 
        
    printf("For file %d there were %d total halos within ionized regions (out of %d halos in this snapshot, a ratio of %.4f). There were %d total halos with a reionization modifier lower than 1.0 (a ratio of %.4f to the total number of halos in this snapshot). The average ionization modifier for these is %.4f\n", filenr, NHalos_In_Regions, NHalos_ThisSnap, (float)NHalos_In_Regions / (float)NHalos_ThisSnap, NHalos_Ionized, (float)NHalos_Ionized / (float)NHalos_ThisSnap, sum_ReionMod / NHalos_Ionized);
          
    status = save_arrays(HaloID, ReionMod, params, NHalos_Ionized, filenr, ThisSnap, first_run);
    if (status != EXIT_SUCCESS)
    {
      ABORT(EXIT_FAILURE);
    }

    free_memory(&TreeNHalos, &HaloID, &ReionMod);

    printf("\n\n"); 
  }

  // Everything done, time to free!

  status = free_grid(Grid);
  if (status != EXIT_SUCCESS)
  {
    ABORT(EXIT_FAILURE);
  }

  free(params); 
  return EXIT_SUCCESS;
} 
