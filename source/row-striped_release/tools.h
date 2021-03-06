#ifndef __MAPPING_H_INCLUDED__
#define __MAPPING_H_INCLUDED__

#include <math.h>
#include <stdlib.h>

#define NROWS 	12			//number of rows in global matrix
#define	NCOLS 	12			//number of columns in global matrix
#define NPROC 	4			//number of processors being used
#define NREPS	1			//number of times to cycle through processors
#define RSIZE	NCOLS			//size of row on locally owned matrix
#define CSIZE	(NROWS/(NPROC*NREPS))	//size of column on locally owned matrix
#define ROOTP	0			//designate any process as root process

/*--
 * ------------------------------------------------------------
 *  ROW STRIPED MAP
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
     * DOCSTRING NEEDS REVISION
     * (i, j) denotes the indices of an element in the unified matrix and (x, y)
     * denotes the indices of an element in the local matrix.
     * The transformation between these two systems occurs as
     * follows : (i, j) <--> (x, y, rank), where 'rank' is the rank of the 
     * process that is in possession of the queried element.
     */
    *owner_rank	= floor ((i * NCOLS + j)/(RSIZE * CSIZE));
    if (caller_rank == *owner_rank) {
	*x	= i - (*owner_rank * CSIZE);
	*y	= j;
	return;
    }
    else if (caller_rank < *owner_rank) {
	*x = -1;
	*y = -1;
	return;
    }
    else {
	//caller_rank > *owner_rank
	*x = 0;
	*y = j;
	return;
    }
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

int disp_formatted (double *matrix, int nproc) {
    /*--
     * Displays the provided matrix with the obvious logical formatting.
     * Further functionality may be added like ROW and COL labelling.
     * NPROC label defaults to -1 if not supplied.
     */
    int i, j, k, ierr=1;

    if (nproc == -1)
	printf ("\nMatrix Output | Owner processor not specified:");
    else
	printf ("\nMatrix Output | Processor=%d", nproc);

    for (k=0; k<NREPS; ++k) {
	printf ("\nSubmatrix=%d", k);
	for (i=0; i<CSIZE; ++i) {
	    printf ("\nRow %d:\t", i+1);
	    for (j=0; j<RSIZE; ++j) {
		printf ("%f \t", matrix[index(k*CSIZE+i, j)]);
	    }
	}
    }
    printf("\n");
    ierr = 0;
    return ierr;
}
//---------------------------------------------------------------------------------

#endif
