#include <stdio.h>
#include "tools.h"
#include "mpi.h"

#define MPI_Assert(X,Y,Z,err)	if (X != Y){ \
    				printf(Z); \
	    			MPI_Abort(MPI_COMM_WORLD,err); }

#define Assert(X,Y,Z)		if (X != Y){ \
    				printf(Z); }

#define MPI_Root(X)		if (rank == ROOTP) { X; }
int main (int argc, char **argv) {
    int i, j, x, y, owner_rank, ierr=0, numtasks, rank, k;
    double *matrix = malloc (NREPS * CSIZE * RSIZE *  sizeof(double));

    MPI_Assert(MPI_Init (&argc, &argv), MPI_SUCCESS, 
	    "Failed to initialize MPI Environment", 101);
    
    MPI_Assert(MPI_Comm_size (MPI_COMM_WORLD, &numtasks), MPI_SUCCESS, 
	    "Error in querying communicator size", 102);

    MPI_Assert(numtasks, NPROC, 
	    "Number of processors specified is not consistent", 501);
    
    MPI_Assert(MPI_Comm_rank (MPI_COMM_WORLD, &rank), MPI_SUCCESS,
	    "Error in querying task ID", 103);

    MPI_Root (printf("Detected %d tasks. Root is %d.\n", numtasks, rank));

    for (i=0; i<CSIZE; ++i)
	for (j=0; j<RSIZE; ++j)
	    matrix[index(i,j)] = i*RSIZE + j;

    for (k=0; k<numtasks; ++k) {
	if (k == rank)
		Assert(disp_formatted(matrix, rank),
			0,"Failed display!");
	MPI_Barrier (MPI_COMM_WORLD);
    }

    double *operating_row = malloc (RSIZE * sizeof(double));
    operating_row = &matrix[index(1,0)];

    for (k=0; k<numtasks; ++k) {
	if (k == rank) {
	    printf ("\n");
	    for (i=0; i<RSIZE; ++i)
		printf ("%f\t",operating_row[i]);
	}
	MPI_Barrier (MPI_COMM_WORLD);
    }




    MPI_Finalize ();
    return 0;
}
