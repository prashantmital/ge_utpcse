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
#include <string.h>
#include "mpi.h"

#define index(x,y)		(x)*RSIZE + y

#define MPI_Assert(X,Y,Z,err)	if (X != Y){ \
    				printf(Z); \
	    			MPI_Abort(MPI_COMM_WORLD,err); }

#define Assert(X,Y,Z)		if (X != Y){ \
    				printf(Z); }

#define MPI_Root(X)		if (rank == ROOTP) { X; }

/*--
 * NROWS 			number of rows in global matrix
 * NCOLS			number of columns in global matrix
 * NPROC			number of processors being used
 * NREPS			number of stripes on each processor
 */
#define RSIZE	NCOLS			//size of row on locally owned matrix
#define CSIZE	(NROWS/(NPROC*NREPS))	//size of column on locally owned matrix
#define ROOTP	0			//designate any process as root process

/*--
 * ------------------------------------------------------------
 * GAUSSIAN ELIMINATION W/ PARTIAL PIVOTING 
 * ------------------------------------------------------------
 */

unsigned int NROWS, NCOLS, NPROC, NREPS;
int parse_command_line (int argc, char *argv[]) {
    int i, ierr = 0;
    if (argc != 9) { printf ("Number of Arguments is %d", argc); ierr = 1; }
    else {
	char *end;
	for (i=1; i<argc;) {
	   if (strcmp(argv[i], "nrows") == 0) {
	       NROWS = strtol (argv[i+1], &end, 10);
	       i += 2;
	   }
	   else if (strcmp(argv[i], "ncols") == 0) {
	       NCOLS = strtol (argv[i+1], &end, 10);
	       i += 2;
	   }
	   else if (strcmp(argv[i], "nproc") == 0) {
	       NPROC = strtol (argv[i+1], &end, 10);
	       i += 2;
	   }
	   else if (strcmp(argv[i], "nreps") ==0) {
	       NREPS = strtol (argv[i+1], &end, 10);
	       i += 2;
	   }
	   else { ierr = 1; break; }
	}
    }
    return ierr; 
}

void help_message () {
    printf ("\nThis program takes command line arguments.\n"
	    "ibrun -np <NPROC> main.out nrows <NROWS>"
	    " ncols <NCOLS> nproc <NPROC> nreps <NREPS>\n");
}

void echo_parameters () {
    printf ("\nNROWS: %d\nNCOLS: %d\nNPROC: %d"
	    "\nNREPS: %d\nRSIZE: %d\nCSIZE: %d\n", 
	    NROWS, NCOLS, NPROC, NREPS, RSIZE, CSIZE);
}

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
int main (int argc, char **argv) {
    //Struct for MPI_Allreduce (MAX_LOC)
    struct {
	double value;
	int rank;
    } local_in, global_out;

    int ierr = 0;
    int i, j, x, y, owner_rank, numtasks, rank, k;
    
    MPI_Init (&argc, &argv); 
    MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    ierr = parse_command_line (argc, argv);
    if (ierr) { 
       MPI_Root (help_message ());
       MPI_Abort (MPI_COMM_WORLD, 000);
    }     
    MPI_Root (echo_parameters ());
    
    double *matrix = malloc (NREPS * CSIZE * RSIZE *  sizeof(double));
    double pp_max; int pp_loc;
    double t1, t2;

    MPI_Assert(numtasks, NPROC, 
	    "Number of processors specified is not consistent", 501);

    MPI_Root (printf("Detected %d tasks. Root is %d.\n", numtasks, rank));

    Assert(init_rand(matrix, rank+1), 0, "Failed initialization!");

    //MPI_Root (printf ("\n Initial Matrix \n--------------------------------------------\n"));
    //for (k=0; k<NREPS; ++k)
	//for (i=0; i<numtasks; ++i) {
	//    if (i == rank)
	//	Assert(disp_formatted(matrix, rank, k),
	//		0,"Failed display!");
	//    MPI_Barrier (MPI_COMM_WORLD);
	//}

    double *operating_row = malloc (1 * RSIZE * sizeof(double));
    int row_start_index, col_start_index, dest, src, iters;
    int tag1 = 351, tag2 = 451, tag3 = 551;
    int pp_row_no=0, pp_col_no, pp_dummy_rank;
    MPI_Status status;

    //Define Derived MPI Datatype for communication
    MPI_Datatype MPI_ROW;
    MPI_Type_contiguous (RSIZE, MPI_DOUBLE, &MPI_ROW);
    MPI_Type_commit (&MPI_ROW);

    //Fix number of iterations and output system type
    iters = fmin (NROWS, NCOLS);
    if (NROWS>NCOLS) {	
	MPI_Root (printf ("\nSystem is over-determined. Row Count = %d, Column Count = %d.\n", NROWS, NCOLS)); }
    else if (NCOLS>NROWS) {
	MPI_Root (printf ("\nSystem is under-determined. Row Count = %d, Column Count = %d.\n", NROWS, NCOLS)); }
    else {
	MPI_Root (printf ("\nSystem is square. Row Count = Column Count = %d.\n", NCOLS)); }

    //Barrier for timing synchronization
    MPI_Barrier (MPI_COMM_WORLD);
    t1 = MPI_Wtime ();

    for (i=0; i<iters; ++i) { 
	global_to_local_map (i,i,rank,&x,&y,&owner_rank);
	local_to_global_map (x,y,rank,&pp_row_no,&pp_col_no,&pp_dummy_rank);
	
	//Partial Pivoting
	get_pivoting_parameters (matrix, x, y, &pp_max, &pp_loc);
	local_in.value = pp_max;
	local_in.rank = rank;
	MPI_Allreduce (&local_in, &global_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

	if (global_out.rank == owner_rank)
	{
	    if (rank == owner_rank)
	    {
		if (x != pp_loc) {
		    //printf ("\nFlag 1 Swapping Rows %d and %d\n", i, pp_row_no);
		    vec_to_vec (&matrix[index(x,0)], &operating_row[0]); 
		    vec_to_vec (&matrix[index(pp_loc,0)], &matrix[index(x,0)]);
		    vec_to_vec (&operating_row[0], &matrix[index(pp_loc,0)]);
		    local_to_global_map (pp_loc, 0, rank, &pp_row_no, &pp_col_no, &pp_dummy_rank); 
		}
	    }
	}
	else {
	    if (rank == global_out.rank) {
		//printf ("\nFlag 2Swapping Rows %d and %d\n", i, pp_row_no);
		MPI_Send (&matrix[index(pp_loc,0)], 1, MPI_ROW, owner_rank, tag1, MPI_COMM_WORLD);
		MPI_Recv (&operating_row[0], 1, MPI_ROW, owner_rank, tag2, MPI_COMM_WORLD, &status);
		vec_to_vec (&operating_row[0], &matrix[index(pp_loc, 0)]);
		local_to_global_map (pp_loc, 0, rank, &pp_row_no, &pp_col_no, &pp_dummy_rank);
	    }
	    else if (rank == owner_rank) {
		MPI_Recv (&operating_row[0], 1, MPI_ROW, global_out.rank, tag1, MPI_COMM_WORLD, &status);
		MPI_Send (&matrix[index(x,0)], 1, MPI_ROW, global_out.rank, tag2, MPI_COMM_WORLD);
		vec_to_vec (&operating_row[0], &matrix[index(x,0)]);
	    }
	}
		
	//MPI_Barrier (MPI_COMM_WORLD);
	//MPI_Root (printf ("\n After Pivoting [i=%d] \n--------------------------------------------\n", i));
	//for (k=0; k<NREPS; ++k)
	//    for (j=0; j<numtasks; ++j) {
	//	if (j == rank)
	//	    Assert(disp_formatted(matrix, rank, k),
	//		    0,"Failed display!");
	//	MPI_Barrier (MPI_COMM_WORLD);
	//    }
	  
	
	if (rank == owner_rank) {
	    //Current processor owns pivot
	    //DEBUG//
	    //printf ("\nMapping:(%d,%d,%d)->(%d,%d,%d)",i,i,rank,x,y,owner_rank);
	    //printf ("\nmatrix[index(x,y)]=matrix[%d]=%f", index(x,y), matrix[index(x,y)]);
	    if (fabs(matrix[index(x,y)])<1E-14) {
		printf ("\nMatrix is singular or very nearly singular. Aborting. "
			"Error encountered while evaluating row %d.\n", i);
		MPI_Abort (MPI_COMM_WORLD, 999);
	    }

	    row_start_index = x+1;
	    col_start_index = y;
	    for (j=0; j<NPROC; ++j) {
		if (j==rank)
		    continue;			//Don't send to self
		dest = j;
		MPI_Send (&matrix[index(x,0)], 1, MPI_ROW, dest, tag3, MPI_COMM_WORLD);
	    }
	    gauss_step (matrix, &matrix[index(x,0)], row_start_index, col_start_index);
	    gauss_scale (matrix, row_start_index-1, col_start_index);
	}
	else if (x!=-1 && y!=-1) {
	    //Current processor has elements belonging to pivot submatrix
	    //DEBUG//
	    //printf ("\nMapping:(%d,%d,%d)->(%d,%d,%d)",i,i,rank,x,y,owner_rank);
	    row_start_index = x;
	    col_start_index = y;
	    src = owner_rank;
	    MPI_Recv (&operating_row[0], 1, MPI_ROW, src, tag3, MPI_COMM_WORLD, &status);
	    int k;
	    gauss_step (matrix, operating_row, row_start_index, col_start_index);
	}
	else {
	    //IDLE: Current processor has no elements in pivot submatrix
	    row_start_index = -1;
	    col_start_index = -1;
	}
	//MPI_Barrier (MPI_COMM_WORLD);
	//MPI_Barrier (MPI_COMM_WORLD);
	//MPI_Root (printf ("\n After Scaling and Elimination [i=%d] \n--------------------------------------------\n", i));
	//for (k=0; k<NREPS; ++k)
	//    for (j=0; j<numtasks; ++j) {
	//	if (j == rank)
	//	    Assert(disp_formatted(matrix, rank, k),
	//		    0,"Failed display!");
	//	MPI_Barrier (MPI_COMM_WORLD);
	//    }
    }

    MPI_Barrier (MPI_COMM_WORLD);
    t2 = MPI_Wtime ();

    //MPI_Root (printf ("\n RREF Matrix \n--------------------------------------------\n"));
    //for (k=0; k<NREPS; ++k)
	//for (i=0; i<numtasks; ++i) {
	//    if (i == rank)
	//	Assert(disp_formatted(matrix, rank, k),
	//		0,"Failed display!");
	//    MPI_Barrier (MPI_COMM_WORLD);
	//}

    MPI_Root(printf("\nReached End of Program\n"));
    MPI_Root(printf("\nGE algorithm took %f seconds", t2-t1));
    MPI_Finalize ();
    return 0;
}

