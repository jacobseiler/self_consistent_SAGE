#ifndef SELFCON_GRID_H
#define SELFCON_GRID_H

// Proto-Types //

// External //
int32_t init_selfcon_grid(void);
int32_t free_selfcon_grid(struct SELFCON_GRID_STRUCT *grid_to_free);
int32_t update_selfcon_grid(struct GALAXY *g, int32_t grid_idx, int32_t snapshot);
int32_t save_selfcon_grid(void);
int32_t determine_nion(struct GALAXY *g, int32_t snapshot, float *Ngamma_HI, float *Ngamma_HeI, float *Ngamma_HeII);
int32_t determine_fesc(struct GALAXY *g, int32_t snapshot, float *fesc_local);
int32_t zero_selfcon_grid(struct SELFCON_GRID_STRUCT *my_grid);
int32_t get_nion_prefix(char *nion_prefix);
#endif
