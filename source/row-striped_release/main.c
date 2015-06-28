/*--
 * Parallel Computing Project
 * Gaussian Elimination
 * Features:
 * 	- Handles dense matrices
 * 	- Implements block cyclic schemes
 * 	- (Row/Column)-Block Cyclic
 * 	- 2D-Block Cyclic
 * Author: Prashant Mital
 * Created: Apr 19, 2015
 * ----------------------------------------
 * Notes
 * ----------------------------------------
 * - Currently working on Column-Block Cyclic
 *   with 2 processors and 2 repetitions
 * - In order to optimally use memory, and get a
 *   contiguous block of data we use a 1-D array
 *   along with some helper functions to simulate
 *   the locally owned array. The helper functions
 *   can basically allow one to change what order
 *   the array is stored in (row/col major or even
 *   some other ordering). These are given in tools.h
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

    MPI_Init (&argc, &argv);
    
    MPI_Comm_size (MPI_COMM_WORLD, &numtasks);

    MPI_Assert(numtasks, NPROC, 
	    "Number of processors specified is not consistent", 501);
    
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    MPI_Root (printf("Detected %d tasks. Root is %d.\n", numtasks, rank));

    Assert(init_rand(matrix, i+rank+1), 0, "Failed initialization!");
    /*for (i=0; i<CSIZE; ++i)
	for (j=0; j<RSIZE; ++j)
	    matrix[index(i,j)] = i*RSIZE + j + 1; //add 1 to avoid 0 in (0,0)*/

    MPI_Root (printf ("\n Initial Matrix \n--------------------------------------------\n"));
    for (i=0; i<numtasks; ++i) {
	if (i == rank)
	    Assert(disp_formatted(matrix, rank),
		    0,"Failed display!");
	MPI_Barrier (MPI_COMM_WORLD);
    }

    /*--
     * Environment to perform GE is now setup. 
     */

    double *operating_row = malloc (1 * RSIZE * sizeof(double));
    int row_start_index, col_start_index, dest, src, iters;
    int tag1 = 351, tag2 = 451, tag3 = 551;
    MPI_Status status;

    //Define Derived MPI Datatypes for communication
    MPI_Datatype MPI_ROW;
    MPI_Type_contiguous (RSIZE, MPI_DOUBLE, &MPI_ROW);
    MPI_Type_commit (&MPI_ROW);

    //Fix number of iterations and output system type
    if (NROWS>NCOLS) {	
	MPI_Root (printf ("\nSystem is over-determined. Row Count = %d, Column Count = %d.\n", NROWS, NCOLS)); }
    else if (NCOLS>NROWS) {
	MPI_Root (printf ("\nSystem is under-determined. Row Count = %d, Column Count = %d.\n", NROWS, NCOLS)); }
    else {
	MPI_Root (printf ("\nSystem is square. Row Count = Column Count = %d.\n", NCOLS)); }
    iters = fmin (NROWS, NCOLS);

    for (i=0; i<iters; ++i) { 
	global_to_local_map (i,i,rank,&x,&y,&owner_rank);
	if (rank == owner_rank) {
	    //Current processor owns pivot
	    //printf ("\nMapping:(%d,%d)->(%d,%d,%d)",i,i,x,y,owner_rank);
	    operating_row = &matrix[index(x,0)];
	    row_start_index = x+1;
	    col_start_index = y;
	    for (j=owner_rank+1; j<NPROC; ++j) {
		dest = j;
		MPI_Send (&operating_row[0], 1, MPI_ROW, dest, tag1, MPI_COMM_WORLD);
	    }
	    gauss_step (matrix, operating_row, row_start_index, col_start_index);
	    gauss_scale (matrix, x, col_start_index);
	}
	else if (x!=-1 && y!=-1) {
	    //Current processor has elements belonging to pivot submatrix
	    row_start_index = x;
	    col_start_index = y;
	    src = owner_rank;
	    MPI_Recv (&operating_row[0], 1, MPI_ROW, src, tag1, MPI_COMM_WORLD, &status);
	    gauss_step (matrix, operating_row, row_start_index, col_start_index);
	}
	else {
	    //IDLE: Current processor has no elements in pivot submatrix
	    row_start_index = -1;
	    col_start_index = -1;
	}
	MPI_Barrier (MPI_COMM_WORLD);
    }

    MPI_Root (printf ("\n RREF Matrix \n--------------------------------------------\n"));
    for (i=0; i<numtasks; ++i) {
	if (i == rank)
	    Assert(disp_formatted(matrix, rank),
		    0,"Failed display!");
	MPI_Barrier (MPI_COMM_WORLD);
    }

    MPI_Root(printf("\nReached End of Program\n"));
    MPI_Finalize ();
    return 0;
}

