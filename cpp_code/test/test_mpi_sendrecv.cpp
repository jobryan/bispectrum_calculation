/*
test_mpi_sendrecv.cpp

Created on May 29, 2014
Updated on June 3, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Testing code for C++

@command:   Needs to be made then run:

            mpic++ test_mpi.cpp -o test_mpi

            then run (e.g., with 4 processors)

            mpirun -np 4 ./test_mpi

*/

// C++ imports
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include <ctime>
#include <math.h>

// 3rd party imports
#include <mpi.h>

using namespace std;

/* globals */
int numnodes,myid,mpi_err;
#define mpi_root 0
#define _USE_MATH_DEFINES
/* end globals */

int main(int i_argc, char* s_argv[])
{

    // MPI Initialization
    
    int i_rank, i_size;
     
    MPI_Init(&i_argc, &s_argv);      /* starts MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);        /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &i_size);        /* get number of processes */
    //printf( "Hello world from process %d of %d\n", i_rank, i_size );
    //MPI_Finalize();
    //return 0;
    
    clock_t o_start;

    if (i_rank == 0){
    
        o_start = clock();
    
    }

    int i_n = 100000000; // best when divisible by i_size

    double d_local_sum = 0.0;

    double d_s = 0.0; // sum for this core
    double d_h = 1.0 / (double)i_n;

    double d_x;

    for (int i=i_rank; i<i_n; i+=i_size){
    
        d_x =d_h * (i + 0.5);
        d_s += 4.0 / (1.0 + pow(d_x,2));
    
    }

    d_local_sum += d_s * d_h;

    // Combine results from all cores

    double d_total_sum; // recvbuffer from Reduce

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&d_local_sum, &d_total_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_root, 
                MPI_COMM_WORLD);

    if (i_rank == 0) {

        double d_error = fabs(d_total_sum - M_PI);
        cout.precision(30);
        cout << "pi is: " << M_PI << endl;
        cout << "pi approximation is: " << d_total_sum << endl;
        cout << "pi is approximately " << d_total_sum << ", error is " 
             << d_error << endl;
        cout << "Time to calculate pi: " 
             << (clock() - o_start) / (double)(CLOCKS_PER_SEC / 1000) 
             << " ms" << endl;
    
    }

    MPI_Finalize();

    return 0;
}
