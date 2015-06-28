#ifndef __MAPPING_H_INCLUDED__
#define __MAPPING_H_INCLUDED__

#include <math.h>
#include <stdlib.h>

#define NROWS 	8			//number of rows in global matrix
#define	NCOLS 	8			//number of columns in global matrix
#define NPROC 	2			//number of processors being used
#define NREPS	1			//number of times to cycle through processors
#define RSIZE	NCOLS			//size of row on locally owned matrix
#define CSIZE	(NROWS/(NPROC*NREPS))	//size of column on locally owned matrix
#define ROOTP	0			//designate any process as root process

/*--
 * ------------------------------------------------------------
 *  ROW CYCLIC MAP
 * ------------------------------------------------------------
 *
 * This file determines the map from the global, unified matrix
 * to the smaller, locally owned chunks located on each
 * individual processor and also provides the functionality 
 * to query, display, modify the data.
 */

//---------------------------------------------------------------------------------

void global_to_local_map (int i, int j, int caller_rank, int *x, int *y, int *owner_rank) {
    /* --
     * This function accepts the global (i,j) location of an element (usually this will
     * be the pivot), as well as the rank of the process which makes the function call.
     * The return values are condition dependent:
     * 	- Scenario 1: The calling process owns the queried element
     * 		In this case x and y will denote the local row_start and col_start
     * 		indices which must be used to begin gaussian elimination. owner_rank will
     * 		equal caller_rank.
     * 	- Scenario 2: The calling process does not own the queried element
     * 		In this case x and y will denote the row_start and col_start indices on the
     * 		locally owned matrices of caller_rank if applicable. In the case that
     * 		cycle_num = NREPS-1 && owner_rank<caller_rank it means that no work needs
     * 		to be done and invalid row and column indices of -1 will be returned
     */

    int block_num, cycle_num;
    
    //Each processor has NREPS blocks of size RSIZE*CSIZE
    block_num 	= floor ((i * NCOLS + j)/(RSIZE * CSIZE));
    //This is the number of processor cycles that have elapsed before the current element
    cycle_num	= floor (block_num / NPROC);

    *owner_rank	= block_num % NPROC;
    //Test for scenarios and assign return values
    if (caller_rank == *owner_rank) {
	*x 	= (cycle_num * CSIZE) + (i % CSIZE);
	*y	= j;
	return;
    }
    else if (caller_rank < *owner_rank) {
	if (cycle_num == NREPS-1) { 
	    *x 	= -1;
	    *y	= -1;
	    return;
	}
	else {
	    *x 	= (cycle_num) * CSIZE;
	    *y	= j;
	    return;
	}
    }
    else {
	// caller_rank > *owner_rank
	*x	= cycle_num * CSIZE;
	*y 	= j;
	return;
    }
}
//---------------------------------------------------------------------------------

void local_to_global_map (int x, int y, int caller_rank, int *i, int *j, int *owner_rank) {
    /* --
     * This is the inverse map from local (x,y) to global (i,j)
     * caller_rank is equal to owner_rank and is included just for consistency
     */
    int cycle_num;

    cycle_num = floor (x/CSIZE);
    *i = cycle_num * CSIZE * NPROC + caller_rank * CSIZE + x % CSIZE;
    *j = y;
    *owner_rank = caller_rank;
}
//---------------------------------------------------------------------------------

int index (int x, int y) {
    /*--
     * Returns the index of local element (x,y) in the linear
     * array that is used to simulate the 2-D case.
     */
    return x*RSIZE + y;			//row major
    //return x*CSIZE +  y;		//column major
}
//---------------------------------------------------------------------------------

int swap_entries (int *A, int *B) {
    int ierr=0, temp;
    temp = A[0];
    A[0] = B[0];
    B[0] = temp;
    return ierr;
}
//---------------------------------------------------------------------------------

int gauss_step (double *matrix, double *operating_row,
	int row_start_index, int col_start_index) {
    /*--
     * Performs an incremental Gauss elimination step with 
     * the given operating row and on the submatrix specified
     * by row_start and col_start indices. 
     */
    int x, y, ierr=0;
    double mult;
    
    for (x = row_start_index; x<CSIZE*NREPS; ++x) {
	mult = matrix[index(x,col_start_index)]/operating_row[col_start_index];
	for (y = col_start_index; y<RSIZE; ++y) {
	    matrix[index(x,y)] = matrix[index(x,y)] - mult * operating_row[y];
	}
    }

    return ierr;
}
//---------------------------------------------------------------------------------

int gauss_scale (double *matrix, int operating_row_index,
	int col_start_index) { 
    /*--
     * Scales the (operating_row_index)th row of matrix by 
     * the logically diagnol element so as to make the 
     * pivot equal unity.
     */
    int y, ierr=0;
    double mult = matrix[index(operating_row_index, col_start_index)];
    matrix[index(operating_row_index,col_start_index)] = 1;
    for (y = col_start_index+1; y<RSIZE; ++y)
       matrix[index(operating_row_index, y)] = 
	   matrix[index(operating_row_index, y)] / mult;

    return ierr;
}
//---------------------------------------------------------------------------------

int get_pivoting_parameters (double *matrix, 
	int operating_row_index, int col_start_index, 
	double *max_local_value, int *max_local_index) {
    /*--
     * Accepts the matrix and the submatrix describing parameters.
     * Traverses the pivot submatrix to determine the row location
     * of the largest local entry and its magnitude
     */    
    *max_local_value = 0;
    *max_local_index = -1;
    int ierr=0;

    if (operating_row_index==-1 || col_start_index==-1) 
	return ierr;

    int i=operating_row_index;
    for (; i<CSIZE*NREPS; ++i) {
	if (fabs(matrix[index(i, col_start_index)]) >= *max_local_value) {
	    *max_local_value = fabs(matrix[index(i, col_start_index)]);
	    *max_local_index = i;
	}
    }
    return ierr;
}
//---------------------------------------------------------------------------------

int vec_to_vec (double *A, double *B) {
    /*--
     * Moves the row that starts at A[0] to the memory
     * addressed by B[0], B[1]... B[RSIZE-1]
     */
    int ierr=0, j;
    for (j=0; j<RSIZE; ++j) {
	B[j] = A[j];
    }
    return ierr;
}
//---------------------------------------------------------------------------------

int init_rand (double *matrix, int seed) {
    /*--
     * Initializes all entried in the given matrix to a random number
     * generated by the rand() function when seeded with the 'seed'
     * argument. Default behavior of rand() when it is called
     * without a prior srand() call is to use a seed of 1. 
     * With the same seed, the rand() function will always generate 
     * the same sequence of numbers.
     */
    int i, j, k, ierr=1;

    srand(seed);
    int normalizer = 10000000;
    for (k=0; k<NREPS; ++k)
	for (i=0; i<CSIZE; ++i) {
	    for (j=0; j<RSIZE; ++j) {
		matrix[index(k*CSIZE+i, j)] = (double)rand()/normalizer;
	}
    }

    ierr = 0;
    return ierr;
}
//---------------------------------------------------------------------------------

int disp_formatted (double *matrix, int nproc, int submtx) {
    /*--
     * Displays the provided matrix with the obvious logical formatting.
     * Further functionality may be added like ROW and COL labelling.
     * NPROC label defaults to -1 if not supplied.
     */
    int i, j, ierr=1;

    if (nproc == -1)
	printf ("\nMatrix Output | Owner processor not specified:");
    else
	printf ("\nMatrix Output | Processor=%d, Submatrix=%d", nproc, submtx);

    for (i=0; i<CSIZE; ++i) {
	printf ("\nRow %d:\t", i+1); 
	for (j=0; j<RSIZE; ++j) {
	    printf ("%f \t", matrix[index(submtx*CSIZE+i, j)]);
	}
    }
    printf("\n");
    ierr = 0;
    return ierr;
}
//---------------------------------------------------------------------------------

#endif
