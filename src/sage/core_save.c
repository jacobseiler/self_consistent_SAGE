#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "core_allvars.h"
#include "core_proto.h"
#include "temporal_array.h"

// keep a static file handle to remove the need to do constant seeking.
FILE* save_fd = NULL;
FILE* save_fd2 = NULL;

void save_galaxies(int filenr, int tree)
{
  char buf[1024];
  int32_t i, max_snap, j;
  
  // now prepare and write galaxies
    // only open the file if it is not already open.
  if (save_fd == NULL)
  {
    sprintf(buf, "%s/%s_z%1.3f_%d", GalaxyOutputDir, RunPrefix, ZZ[ListOutputSnaps[0]], filenr);

    save_fd = fopen(buf, "wb");
    if (save_fd == NULL)
    {
      printf("can't open file `%s'\n", buf);
      ABORT(0);
    }

    int32_t number_header_values = 5 + (6+MAXSNAPS)*2 + Ntrees;
    // write out placeholders for the header data.
    int32_t *tmp_buf = calloc(number_header_values, sizeof(int32_t));
    fwrite(tmp_buf, sizeof(int), number_header_values, save_fd);
    free(tmp_buf);
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
      write_temporal_arrays(&HaloGal[i], save_fd);

      for(int32_t snap = 0; snap < MAXSNAPS; ++snap) {
        if(HaloGal[i].MUV[snap] > -0.1 && HaloGal[i].MUV[snap] < 0.1) {
            fprintf(stderr, "Before Save\tSnap %d has MUV %.4e\n", snap, HaloGal[i].MUV[snap]);
        }
      }
      fwrite(HaloGal[i].MUV, sizeof(float), MAXSNAPS, save_fd);
      TotGalaxies++;
      TreeNgals[tree]++;
    } 
  
  } // NumGals loop.
}

void finalize_galaxy_file(void)
{
  // file must already be open.
  assert(save_fd);

  // seek to the beginning.
  fseek(save_fd, 0, SEEK_SET );

  int32_t steps = STEPS;

  myfwrite(&steps, sizeof(int32_t), 1, save_fd); 
  myfwrite(&MAXSNAPS, sizeof(int32_t), 1, save_fd);
  myfwrite(ZZ, sizeof(*(ZZ)), MAXSNAPS, save_fd); 
  myfwrite(&Hubble_h, sizeof(double), 1, save_fd);
  myfwrite(&Omega, sizeof(double), 1, save_fd);
  myfwrite(&OmegaLambda, sizeof(double), 1, save_fd);
  myfwrite(&BaryonFrac, sizeof(double), 1, save_fd);
  myfwrite(&PartMass, sizeof(double), 1, save_fd);
  myfwrite(&BoxSize, sizeof(double), 1, save_fd);
  myfwrite(&GridSize, sizeof(int32_t), 1, save_fd);
  myfwrite(&Ntrees, sizeof(int32_t), 1, save_fd); 
  myfwrite(&TotGalaxies, sizeof(int), 1, save_fd);
  myfwrite(TreeNgals, sizeof(int), Ntrees, save_fd);

  // close the file and clear handle after everything has been written
  fclose(save_fd);
  save_fd = NULL;
  
}


void save_merged_galaxies(int filenr, int tree)
{
  char buf[1000];
  int i;
   
  if(!save_fd2)
  { 
    sprintf(buf, "%s/%s_MergedGalaxies_%d", GalaxyOutputDir, RunPrefix, filenr);

    save_fd2 = fopen(buf, "wb");
    if (save_fd2 == NULL)
    {
      printf("can't open file `%s'\n", buf);		
      ABORT(0);
    }

    int32_t number_header_values = 5 + (6+MAXSNAPS)*2 + Ntrees;
    // write out placeholders for the header data.
    int32_t *tmp_buf = calloc(number_header_values, sizeof(int32_t));
    fwrite(tmp_buf, sizeof(int), number_header_values, save_fd2);

    // write out placeholders for the header data.
    free(tmp_buf);
  }

  for(i = 0; i < MergedNr; i++)
  {
    write_temporal_arrays(&MergedGal[i], save_fd2);

    for(int32_t snap = 0; snap < MAXSNAPS; ++snap) {
      if(MergedGal[i].MUV[snap] > -0.1 && MergedGal[i].MUV[snap] < 0.1) {
        fprintf(stderr, "Before Merge Save\tSnap %d has MUV %.4e\n", snap, MergedGal[i].MUV[snap]);
      }
    }
    fwrite(MergedGal[i].MUV, sizeof(float), MAXSNAPS, save_fd2);
    //write_temporal_arrays(&MergedGal[i], save_fd2);

    TotMerged++;
    TreeNMergedgals[tree]++;
  }
}

void finalize_merged_galaxy_file(void)
{

  // file must already be open.
  assert(save_fd2);

  // seek to the beginning.
  fseek(save_fd2, 0, SEEK_SET);

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
  fclose(save_fd2);
  save_fd2 = NULL;
  
}
