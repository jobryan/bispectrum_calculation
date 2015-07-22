/*
test_mpi_sendrecv2.cpp

Created on June 10, 2014
Updated on June 10, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Testing code for C++

@command:   Needs to be made then run:

            /home/jobryan/bin/mpic++ test_mpi_sendrecv2.cpp -o test_mpi_sendrecv2 -std=c++11

            then run (e.g., with 4 processors)

            /home/jobryan/bin/mpirun -np 2 ./test_mpi_sendrecv2

*/

// C++ imports
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include <ctime>
#include <math.h>

// 3rd party imports
#include <mpi.h>

using namespace std;

/* Globals */

#define WORKTAG 1
#define DIETAG 2


/* Local functions */

static void master(void);
static void slave(int myrank);
static vector<int> get_next_work_item(void);
static void process_results(vector<double> result);
static vector<double> do_work(vector<int> work);


int
main(int argc, char **argv)
{
  int myrank;

  /* Initialize MPI */

  MPI_Init(&argc, &argv);

  /* Find out my identity in the default communicator */

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if (myrank == 0) {
    master();
  } else {
    slave(myrank);
  }

  /* Shut down MPI */

  MPI_Finalize();
  return 0;
}


static void
master(void)
{
  int ntasks, rank;
  vector<int> work (2,0);
  vector<double> result (2,0);
  //int work = 0;
  //int result;
  MPI_Status status;

  /* Find out how many processes there are in the default
     communicator */

  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

  /* Seed the slaves; send one unit of work to each slave. */

  for (rank = 1; rank < ntasks; ++rank) {

    /* Find the next item of work to do */

    //work = get_next_work_item();
    cout << "Sending initial job..." << endl;
    /* Send it to each rank */

    MPI_Send(&work[0],             /* message buffer */
             work.size(),                 /* one data item */
             MPI_INT,           /* data item is an integer */
             rank,              /* destination process rank */
             WORKTAG,           /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */
  }

  /* Loop over getting new work requests until there is no more work
     to be done */

  cout << "Entering long loop..." << endl;

  int i_job_count = 0;
  //work = get_next_work_item();
  while (i_job_count < 10) {

    /* Receive results from a slave */

    MPI_Recv(&result[0],           /* message buffer */
             result.size(),                 /* one data item */
             MPI_DOUBLE,        /* of type double real */
             MPI_ANY_SOURCE,    /* receive from any sender */
             MPI_ANY_TAG,       /* any type of message */
             MPI_COMM_WORLD,    /* default communicator */
             &status);          /* info about the received message */

    cout << "result: " << result[0] << "," << result[1] << endl;

    /* Send the slave a new work unit */

    MPI_Send(&work[0],             /* message buffer */
             work.size(),                 /* one data item */
             MPI_INT,           /* data item is an integer */
             status.MPI_SOURCE, /* to who we just received from */
             WORKTAG,           /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */

    /* Get the next unit of work to be done */

    //work = get_next_work_item();

    i_job_count++;
  }

  /* There's no more work to be done, so receive all the outstanding
     results from the slaves. */

  cout << "final round up..." << endl;

  for (rank = 1; rank < ntasks; ++rank) {
    MPI_Recv(&result[0], result.size(), MPI_DOUBLE, MPI_ANY_SOURCE,
             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    cout << "result: " << result[0] << "," << result[1] << endl;
  }

  /* Tell all the slaves to exit by sending an empty message with the
     DIETAG. */

  for (rank = 1; rank < ntasks; ++rank) {
    MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
  }
}


static void 
slave(int myrank)
{

  vector<int> work (2,0);
  vector<double> result (2,1);
  //int work;
  //int result = 1;
  MPI_Status status;

  cout << "Hello from slave " << myrank << endl;

  while (1) {

    /* Receive a message from the master */

    MPI_Recv(&work[0], work.size(), MPI_INT, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);

    /* Check the tag of the received message. */

    if (status.MPI_TAG == DIETAG) {
      return;
    }

    /* Do the work */

    //result = do_work(work);

    /* Send the result back */

    //MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&result[0], result.size(), MPI_DOUBLE, 0, 1, 
                MPI_COMM_WORLD);
  }
}


static vector<int> 
get_next_work_item(void)
{
  /* Fill in with whatever is relevant to obtain a new unit of work
     suitable to be given to a slave. */

  vector<int> vi_work(3,0);

  return vi_work;

}


static void 
process_results(vector<double> result)
{
  /* Fill in with whatever is relevant to process the results returned
     by the slave */
}


static vector<double>
do_work(vector<int> work)
{
  /* Fill in with whatever is necessary to process the work and
     generate a result */

  //int i_rand = rand();

}

/*
Equivalent to Python's xrange(first, last, inc)
*/

vector<int> xrange(int i_first, int i_last, int i_incrememt=1){

    vector<int> vi_return;

    for (int i=i_first; i<i_last+1; i+=i_incrememt){

        vi_return.push_back(i);

    }

    return vi_return;

}
