EXEC   = grid_sage 

OBJS   = 	./main_grid.o \
			./core_init_grid.o \
			./core_allvars_grid.o \
			./core_read_parameter_file_grid.o \
			./core_io_gals.o \
			./core_update_grid.o \
			./core_save_grid.o \
			./core_mymalloc.o

INCL   =	./core_allvars_grid.h  \
			./core_proto_grid.h \
			./Makefile 

#USE-MPI = YES

ifdef USE-MPI
    OPT += -DMPI  #  This creates an MPI version that can be used to process files in parallel
    CC = mpicc  # sets the C-compiler
else
    CC = gcc  # sets the C-compiler
endif

#OPT += -DDEBUG_GALS 

GSL_INCL = -I/usr/local/include  # make sure your system knows where GSL_DIR is
GSL_LIBS = -L/usr/local/lib
OPTIMIZE = -g -O0 -Wall  # optimization and warning flags

LIBS   =   -g -lm  $(GSL_LIBS) -lgsl -lgslcblas 

CFLAGS =   $(OPTIONS) $(OPT) $(OPTIMIZE) $(GSL_INCL)

default: all

$(EXEC): $(OBJS) 
	$(CC) $(OPTIMIZE) $(OBJS) $(LIBS)   -o  $(EXEC) 

$(OBJS): $(INCL) 

clean:
	rm -f $(OBJS)

tidy:
	rm -f $(OBJS) ./$(EXEC)

all:  $(EXEC) 

celan celna clena claen:clean
