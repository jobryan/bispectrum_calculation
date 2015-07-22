'''
mpi_loop_example.py

Created on May 21, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Code to simply demonstrate how to use MPI to distribute loop 
            calculations to multiple processors. Important to execute this 
            code with mpiexec as follows (in this example, 5 cores are used):

            mpiexec -n 5 python mpi_loop_example.py

            This example computes pi using

            pi = int[ 4 / (1 + x**2), 0, 1] 
               ~ 1 / n * sum[ 4 / (1 + (((i + 0.5) / n)**2 ), i = (0, n-1) ]

            For alternative (more standard) approaches to this distribution, 
            see examples at 

            http://www.bu.edu/pasi/files/2011/01/Lisandro-Dalcin-mpi4py.pdf

            around slide 38.

            The key here is in the loop: we iterate using

            range(i_rank, i_n, i_size)

            which gives us the array of points from i_rank (e.g., 0 for the 
            first core), to i_n (number of steps), spaced i_size (e.g., 5 if you
            are running the process with 5 cores) apart. Since the i_rank will
            be different on each core, you will uniquely cover the range.

            Additionally, we need to remember to collect the results from each 
            of the cores. First, we stop the program from running so that 
            later calculations don't run before the parallelized code finishes.
            This is done with

            o_comm.Barrier()

            Next, we gather all of the results from each core into one place.
            This is done with

            o_comm.Reduce(send_buffer, receive_buffer, op=MPI.SUM)

            where the send_buffer and receive_buffer objects need to be packaged
            in numpy arrays even if they are floats (as they are in this case).
            The syntax for this highlights the need for explicitly typing the 
            arrays (e.g., np.array( ..., dtype='d')) as well as the type of data
            that should be read from the buffer (e.g., MPI.DOUBLE).

            *NOTE: All code here is written utilizing Hungarian notation for
            clarity (which may come at the cost of brevity in some cases). For 
            more details, see

            http://wiki.quantsoftware.org/index.php?title=Coding_Standard

@inputs:    None

@outputs:   Prints and plots results of sum -- in this case

'''

# Python imports
import time

# 3rd party imports
import numpy as np
from mpi4py import MPI

'''
MPI Initialization
'''
o_comm = MPI.COMM_WORLD
i_rank = o_comm.Get_rank() # current core number -- e.g., i in arange(i_size)
i_size = o_comm.Get_size() # number of cores assigned to run this program

'''
Multiple loop summation: Calculating pi
'''

# Calculation of pi

if i_rank == 0:
    f_t1 = time.time()

i_n = 100000000 # best when divisible by i_size

f_local_sum = 0.0

f_s = 0.0 # sum for this core
f_h = 1.0 / i_n

for i_ in range(i_rank, i_n, i_size):
    f_x = f_h * (i_ + 0.5)
    f_s += 4.0 / (1.0 + f_x**2)

f_local_sum += f_s * f_h

# Combine results from all cores

f_total_sum = np.array(0., dtype='d') # recvbuffer from Reduce--must be na type

o_comm.Barrier()
o_comm.Reduce([np.array(f_local_sum, dtype='d'), MPI.DOUBLE], 
                [f_total_sum, MPI.DOUBLE], op=MPI.SUM)

if i_rank == 0:
    f_t2 = time.time()
    f_error = abs(f_total_sum - np.pi)
    print "pi is approximately %.16f, error is %.16f" % (f_total_sum, f_error)
    print "time per core: %.3f, for n = %i" % (f_t2 - f_t1, i_n)