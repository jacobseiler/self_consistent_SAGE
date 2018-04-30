#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "core_allvars.h"
#include "core_proto.h"

// keep a static file handle to remove the need to do constant seeking.
FILE* save_fd[ABSOLUTEMAXSNAPS] = { 0 };
FILE* save_fd2 = NULL;

void save_galaxies(int filenr, int tree)
{
  char buf[1024];
  int32_t i, n, max_snap, j;

  int OutputGalCount[MAXSNAPS], *OutputGalOrder;
  
  OutputGalOrder = (int*)malloc( NumGals*sizeof(int) );
  assert( OutputGalOrder );

  // reset the output galaxy count and order
  for(i = 0; i < MAXSNAPS; i++)
    OutputGalCount[i] = 0;
  for(i = 0; i < NumGals; i++)
    OutputGalOrder[i] = -1;

  // first update mergeIntoID to point to the correct galaxy in the output
  for(n = 0; n < NOUT; n++)
  {
    for(i = 0; i < NumGals; i++)
    {
      if(HaloGal[i].SnapNum == ListOutputSnaps[n])
      {
        OutputGalOrder[i] = OutputGalCount[n];
        OutputGalCount[n]++;
      }
    }
  }
    
  for(i = 0; i < NumGals; i++)
    if(HaloGal[i].mergeIntoID > -1)
      HaloGal[i].mergeIntoID = OutputGalOrder[HaloGal[i].mergeIntoID];    
  
  // now prepare and write galaxies
  for(n = 0; n < NOUT; n++)
  {    
    // only open the file if it is not already open.
    if( !save_fd[n] )
    {
      sprintf(buf, "%s/%s_z%1.3f_%d", OutputDir, FileNameGalaxies, ZZ[ListOutputSnaps[n]], filenr);

      save_fd[n] = fopen(buf, "wb");
      if (save_fd == NULL)
      {
				printf("can't open file `%s'\n", buf);
				ABORT(0);
      }

      int32_t number_header_values = 5 + (6+MAXSNAPS)*2 + Ntrees;
			// write out placeholders for the header data.
			int32_t *tmp_buf = malloc(number_header_values * sizeof(int32_t));
			memset( tmp_buf, 0, number_header_values * sizeof(int32_t));
			fwrite( tmp_buf, sizeof(int), number_header_values, save_fd[n] );
			free( tmp_buf );
		}
    
    // There are some galaxies that aren't at the root redshift but do not have any descendants.
    // This block of code catches these excepetions.
    // A more detailed comment on this is located in the 'free_galaxies_and_tree' function in 'core_io_tree.c'. 
 
    for(i = 0; i < NumGals; i++)
    {
      max_snap = 0;

      if(HaloGal[i].IsMerged != -1)
        continue;
      for(j = 1; j < MAXSNAPS; ++j)
      {
        if(HaloGal[i].GridHistory[j] != -1)
        {    
          max_snap = j;
        } 
      }

      if(HaloGal[i].SnapNum == max_snap && Halo[HaloGal[i].HaloNr].Descendant == -1) 
      {
        write_temporal_arrays(&HaloGal[i], save_fd[n]); // Input snapshots are ordered highest -> lowest so it'll be 0th element. 
        TotGalaxies[n]++;
        TreeNgals[n][tree]++;          
      } 
    
    } // NumGals loop.
  } // NOUT loop.

  // don't forget to free the workspace.
  free( OutputGalOrder );

}

void finalize_galaxy_file(void)
{
  int n;

  for(n = 0; n < NOUT; n++)
  {
    // file must already be open.
    assert( save_fd[n] );

    // seek to the beginning.
    fseek( save_fd[n], 0, SEEK_SET );
  
    int32_t steps = STEPS;
 
    myfwrite(&steps, sizeof(int32_t), 1, save_fd[n]); 
    myfwrite(&MAXSNAPS, sizeof(int32_t), 1, save_fd[n]);
    myfwrite(ZZ, sizeof(*(ZZ)), MAXSNAPS, save_fd[n]); 
    myfwrite(&Hubble_h, sizeof(double), 1, save_fd[n]);
    myfwrite(&Omega, sizeof(double), 1, save_fd[n]);
    myfwrite(&OmegaLambda, sizeof(double), 1, save_fd[n]);
    myfwrite(&BaryonFrac, sizeof(double), 1, save_fd[n]);
    myfwrite(&PartMass, sizeof(double), 1, save_fd[n]);
    myfwrite(&BoxSize, sizeof(double), 1, save_fd[n]);
    myfwrite(&GridSize, sizeof(int32_t), 1, save_fd[n]);
    myfwrite(&Ntrees, sizeof(int32_t), 1, save_fd[n]); 
    myfwrite(&TotGalaxies[n], sizeof(int), 1, save_fd[n]);
    myfwrite(TreeNgals[n], sizeof(int), Ntrees, save_fd[n]);

    // close the file and clear handle after everything has been written
    fclose( save_fd[n] );
    save_fd[n] = NULL;
  }
  
}


void save_merged_galaxies(int filenr, int tree)
{
  char buf[1000];
  int i;
  //struct GALAXY_OUTPUT galaxy_output;

  int OutputGalCount, *OutputGalOrder;
  
  OutputGalOrder = (int*)malloc( MergedNr*sizeof(int) );
  assert( OutputGalOrder );

  // reset the output galaxy count and order 
  OutputGalCount = 0;
  for(i = 0; i < MergedNr; i++)
  {
    OutputGalOrder[i] = i;
    OutputGalCount++;
  }

  if( !save_fd2 )
  { 
    sprintf(buf, "%s/%s_MergedGalaxies_%d", OutputDir, FileNameGalaxies, filenr);

    save_fd2 = fopen(buf, "wb");
    if (save_fd2 == NULL)
    {
      printf("can't open file `%s'\n", buf);		
      ABORT(0);
    }


    int32_t number_header_values = 5 + (6+MAXSNAPS)*2 + Ntrees;
    // write out placeholders for the header data.
    int32_t *tmp_buf = malloc(number_header_values * sizeof(int32_t));
    memset( tmp_buf, 0, number_header_values * sizeof(int32_t));
    fwrite( tmp_buf, sizeof(int), number_header_values, save_fd2);

    // write out placeholders for the header data.
    free( tmp_buf );
  }

  for(i = 0; i < MergedNr; i++)
  {
    write_temporal_arrays(&MergedGal[i], save_fd2);

    TotMerged++;
    TreeNMergedgals[tree]++;
  }

  // don't forget to free the workspace.
  free( OutputGalOrder );

}

void finalize_merged_galaxy_file(void)
{

  // file must already be open.
  assert( save_fd2 );

  // seek to the beginning.
  fseek( save_fd2, 0, SEEK_SET );

  int32_t steps = STEPS;
 
  myfwrite(&steps, sizeof(int32_t), 1, save_fd2); 
  myfwrite(&MAXSNAPS, sizeof(int32_t), 1, save_fd2);
  myfwrite(ZZ, sizeof(*(ZZ)), MAXSNAPS, save_fd2); 
  myfwrite(&Hubble_h, sizeof(double), 1, save_fd2);
  myfwrite(&Omega, sizeof(double), 1, save_fd2);
  myfwrite(&OmegaLambda, sizeof(double), 1, save_fd2);
  myfwrite(&BaryonFrac, sizeof(double), 1, save_fd2);
  myfwrite(&PartMass, sizeof(double), 1, save_fd2);
  myfwrite(&BoxSize, sizeof(double), 1, save_fd2);
  myfwrite(&GridSize, sizeof(int32_t), 1, save_fd2);

  myfwrite(&Ntrees, sizeof(int), 1, save_fd2);
  myfwrite(&TotMerged, sizeof(int), 1, save_fd2);
  myfwrite(TreeNMergedgals, sizeof(int), Ntrees, save_fd2);

  // close the file and clear handle after everything has been written
  fclose( save_fd2 );
  save_fd2 = NULL;
  
}
