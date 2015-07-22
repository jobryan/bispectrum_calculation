'''
cl21.py

Input is either map_data or map_sim; output is cl21_data or cl21_sim, 
respectively.

@command:   ** Needs to run on elgordo due to strange MPI (slash mpi4py) issues
            on cirrus.

            mpirun -np 12 python -W ignore cl21.py data 0
            mpirun -np 12 python -W ignore cl21.py sim i

            where i is the sim number

'''

# Python imports
import time
import pickle
import itertools as it
import sys

# 3rd party imports
import numpy as np
import healpy as hp
from mpi4py import MPI

#
import helpers as h

'''
Main: Default run
'''

def main(run_type='data', nsim=0, fnl=0):

    '''
    MPI Setup
    '''
    o_comm = MPI.COMM_WORLD
    i_rank = o_comm.Get_rank() # current core number -- e.g., i in arange(i_size)
    i_size = o_comm.Get_size() # number of cores assigned to run this program
    o_status = MPI.Status()

    i_work_tag = 0
    i_die_tag = 1

    '''
    Loading and calculating power spectrum components
    '''

    if (run_type == 'fnl'):
        nl = 1024
    else:
        nl = 1499

    if (run_type == 'data'):
        fn_map = h._fn_map
    elif (run_type == 'sim'):
        fn_map = 'output/map_sim_%i.fits' % nsim
    elif (run_type == 'fnl'):
        #print "fnl value: %i" % fnl
        fn_map = 'data/fnl_sims/map_fnl_%i_sim_%i.fits' % (int(fnl), nsim)

    # (1) read map (either map_data or map_sim), mask, mll, and create mll_inv;
    map_in = hp.read_map(fn_map)
    mask = hp.read_map(h._fn_mask)
    if (run_type == 'fnl'):
        mask = 1.
    fn_mll = 'output/na_mll_%i_lmax.npy' % nl
    mll = np.load(fn_mll)
    if (run_type == 'fnl'):
        mll = np.identity(nl)
    mll_inv = np.linalg.inv(mll)

    nside = hp.get_nside(map_in)

    if (i_rank == 0):

        if (run_type == 'data'):
            fn_cl21 = 'output/cl21_data.dat'
            fn_cl21_no_mll = 'output/cl21_data_no_mll.dat'
        elif (run_type == 'sim'):
            fn_cl21 = 'output/cl21_sim_%i.dat' % nsim
            fn_cl21_no_mll = 'output/cl21_no_mll_%i.dat' % nsim
        elif (run_type == 'fnl'):
            fn_cl21 = 'output/cl21_fnl_%i_sim_%i.dat' % (int(fnl), nsim)
            fn_cl21_no_mll = 'output/cl21_fnl_%i_sim_%i_no_mll.dat' % (int(fnl), nsim)

        f_t1 = time.time()

        print ""
        print "Run parameters:"
        print "(Using %i cores)" % i_size
        print "nl: %i, nside: %i, map: %s" % (nl, nside, fn_map)
        print "beam: %s, alpha_beta: %s, cltt: %s" % (h._fn_beam, h._fn_alphabeta, h._fn_cltt)

        print ""
        print "Loading ell, r, dr, alpha, beta, cltt, and beam..."

    # (2) normalize, remove mono-/dipole, and mask map to create map_masked;
    map_in /= (1e6 * 2.7)
    map_in = hp.remove_dipole(map_in)
    map_masked = map_in * mask

    # (3) create alm_masked (map2alm on map_masked), cltt_masked (anafast on 
    #     map_masked), and cltt_corrected (dot cltt_masked with mll_inv)

    if (run_type == 'data' or run_type == 'sim'):
        alm_masked = hp.map2alm(map_masked)
    elif (run_type == 'fnl'):
        fn_almg = ('data/fnl_sims/alm_l_%04d_v3.fits' % (nsim,))
        almg = hp.read_alm(fn_almg)
        fn_almng = ('data/fnl_sims/alm_nl_%04d_v3.fits' % (nsim,))
        almng = hp.read_alm(fn_almng)
        alm = almg + fnl * almng
        alm_masked = alm

    cltt_masked = hp.anafast(map_masked)
    cltt_masked = cltt_masked[:nl]
    cltt_corrected = np.dot(mll_inv, cltt_masked)

    # stuff with alpha, beta, r, dr, and beam

    l, r, dr, alpha, beta = np.loadtxt(h._fn_alphabeta, 
                                usecols=(0,1,2,3,4), unpack=True, skiprows=3)

    l = np.unique(l)
    r = np.unique(r)[::-1]

    nr = len(r)

    if (run_type == 'data' or run_type == 'sim'):
        cltt_denom = np.load('output/cltt_theory.npy') #replace with 'output/na_cltt.npy'
        cltt_denom = cltt_denom[:nl]
    elif (run_type == 'fnl'):
        cltt_denom = np.loadtxt('joe/cl_wmap5_bao_sn.dat', usecols=(1,), unpack=True)

    alpha = alpha.reshape(len(l), nr)
    beta = beta.reshape(len(l), nr)
    dr = dr.reshape(len(l), nr)
    dr = dr[0]
    
    if (run_type != 'fnl'):
        beam = np.load(h._fn_beam)
    else:
        #beam = np.ones(len(cltt_denom))
        beam = np.load(h._fn_beam)
        noise = np.zeros(len(cltt_denom))
        nlm = hp.synalm(noise, lmax=nl)

    ####### TEMPORARY -- change beam #######
    #beam = np.ones(len(cltt_denom))
    #noise = np.zeros(len(cltt_denom))
    #nlm = hp.synalm(noise, lmax=nl)
    ########################################

    l = l[:nl]
    beam = beam[:nl]
    alpha = alpha[:nl,:]
    beta = beta[:nl,:]

    if (i_rank == 0):
        print "nr: %i, nl: %i" % (nr, nl)

    f_t2 = time.time()

    if (i_rank == 0):
        print ""
        print "Time to create alms from maps: %.2f s" % (f_t2 - f_t1)
        print "Calculating full skewness power spectrum..."

    cl21 = np.zeros(nl)
    work = np.zeros(1, dtype='i')
    result = np.zeros(nl, dtype='d')

    # master loop
    if (i_rank == 0):

        # send initial jobs

        for i_rank_out in range(1,i_size):

            work = np.array([i_rank_out-1], dtype='i')
            o_comm.Send([work, MPI.INT], dest=i_rank_out, tag=i_work_tag)

        for i_r in range(i_size-1,nr):

            if (i_r % (nr / 10) == 0):
                print "Finished %i%% of jobs... (%.2f s)" % (i_r * 100 / nr,
                time.time() - f_t2)

            work = np.array([i_r], dtype='i')

            o_comm.Recv([result, MPI.DOUBLE], source=MPI.ANY_SOURCE, 
                status=o_status, tag=MPI.ANY_TAG)

            #print "received results from core %i" % o_status.Get_source()

            o_comm.Send([work,MPI.INT], dest=o_status.Get_source(), 
                tag=i_work_tag)

            cl21 += result

        for i_rank_out in range(1,i_size):

            o_comm.Recv([result, MPI.DOUBLE], source=MPI.ANY_SOURCE,
                status=o_status, tag=MPI.ANY_TAG)

            cl21 += result
            print "cl21 = %.6f, result = %.6f" % (np.average(cl21), np.average(result))

            o_comm.Send([np.array([9999], dtype='i'), MPI.INT], 
                dest=o_status.Get_source(), tag=i_die_tag)

    #slave loop:
    else:

        while(1):

            o_comm.Recv([work, MPI.INT], source=0, status=o_status, 
                tag=MPI.ANY_TAG)

            if (o_status.Get_tag() == i_die_tag):

                break

            i_r = work[0]
            #print ("i_r: %i" % i_r)
            #print ("alm_masked: ", alm_masked)
            #print ("cltt_denom: ", cltt_denom)
            #print ("beam: ", beam)
            #print ("mask: ", mask)
            #print ("fn_map: ", fn_map)

            # create Alm, Blm (almxfl with alm_masked and beta / cltt_denom 
            # * beam, etc.)

            Alm = np.zeros(alm_masked.shape[0],complex)
            Blm = np.zeros(alm_masked.shape[0],complex)
            clAB2 = np.zeros(nl+1)
            clABB = np.zeros(nl+1)

            #Alm = hp.almxfl(alm_masked, alpha[:,i_r] / cltt_denom * beam)
            #Blm = hp.almxfl(alm_masked, beta[:,i_r] / cltt_denom * beam)
            #for li in xrange(2,nl):
            #    I = hp.Alm.getidx(nl,li,np.arange(min(nl,li)+1))
            #    Alm[I]=alpha[li-2][i_r]*(alm_masked[I]*beam[li]+nlm[I])/(cltt_denom[li]*beam[li]**2+noise[li])
            #    Blm[I]=beta[li-2][i_r]*(alm_masked[I]*beam[li]+nlm[I])/(cltt_denom[li]*beam[li]**2+noise[li])
            if (run_type == 'fnl'):
                for li in xrange(2,nl):
                    I = hp.Alm.getidx(nl,li,np.arange(min(nl,li)+1))
                    Alm[I]=alpha[li-2][i_r]*(alm_masked[I]*beam[li]+nlm[I])/(cltt_denom[li]*beam[li]**2+noise[li])
                    Blm[I]=beta[li-2][i_r]*(alm_masked[I]*beam[li]+nlm[I])/(cltt_denom[li]*beam[li]**2+noise[li])
            else:
                for li in xrange(2,nl):
                    I = hp.Alm.getidx(nl,li,np.arange(min(nl,li)+1))
                    Alm[I]=alpha[li-2][i_r]*(alm_masked[I])/cltt_denom[li]*beam[li]
                    Blm[I]=beta[li-2][i_r]*(alm_masked[I])/cltt_denom[li]*beam[li]

            ############################# DEBUG ################################
            if i_r == 0:
                cltt_Alm = hp.alm2cl(Alm)
                cltt_Blm = hp.alm2cl(Blm)
                np.savetxt('debug2/cltt_%s_Alm.dat' % run_type, cltt_Alm)
                np.savetxt('debug2/cltt_%s_Blm.dat' % run_type, cltt_Blm)
            ####################################################################

            #An = hp.alm2map(Alm, nside=nside, fwhm=0.00145444104333,
            #    verbose=False)
            #Bn = hp.alm2map(Blm, nside=nside, fwhm=0.00145444104333,
            #    verbose=False)
            An = hp.alm2map(Alm, nside=nside)
            Bn = hp.alm2map(Blm, nside=nside)

            ############################# DEBUG ################################
            if i_r == 0:
                cltt_An = hp.anafast(An)
                cltt_Bn = hp.anafast(Bn)
                np.savetxt('debug2/cltt_%s_An.dat' % run_type, cltt_An)
                np.savetxt('debug2/cltt_%s_Bn.dat' % run_type, cltt_Bn)
            ####################################################################

            An = An * mask
            Bn = Bn * mask

            ############################# DEBUG ################################
            #if i_r == 0:
            #    print "saving alpha, beta for %i" % i_r
            #    np.savetxt('debug2/alpha_ir_%i' % i_r, alpha[:,i_r])
            #    np.savetxt('debug2/beta_ir_%i' % i_r, beta[:,i_r])
            #    print "(An * Bn)[:10] == An[:10] * Bn[:10]:", (An * Bn)[:10] == An[:10] * Bn[:10]
            
            ####################################################################

            B2lm = hp.map2alm(Bn*Bn, lmax=nl)
            ABlm = hp.map2alm(An*Bn, lmax=nl)

            ############################# DEBUG ################################
            if i_r == 0:
                cltt_B2lm = hp.alm2cl(B2lm)
                cltt_ABlm = hp.alm2cl(ABlm)
                np.savetxt('debug2/cltt_%s_B2lm.dat' % run_type, cltt_B2lm)
                np.savetxt('debug2/cltt_%s_ABlm.dat' % run_type, cltt_ABlm)
            ####################################################################

            #clAB2 = hp.alm2cl(Alm, B2lm, lmax=nl)
            #clABB = hp.alm2cl(ABlm, Blm, lmax=nl)
            for li in xrange(2,nl+1):
                I = hp.Alm.getidx(nl,li,np.arange(min(nl,li)+1))
                clAB2[li] = (Alm[I[0]]*B2lm[I[0]].conj()
                        +2.*sum(Alm[I[1:]]*B2lm[I[1:]].conj()))/(2.0*li+1.0)
                clABB[li] = (Blm[I[0]]*ABlm[I[0]].conj()
                        +2.*sum(Blm[I[1:]]*ABlm[I[1:]].conj()))/(2.0*li+1.0)


            ############################# DEBUG ################################
            if i_r == 0:
                np.savetxt('debug2/clAB2_%s.dat' % run_type, clAB2)
                np.savetxt('debug2/clABB_%s.dat' % run_type, clABB)
            ####################################################################

            clAB2 = clAB2[1:]
            clABB = clABB[1:]

            result = np.zeros(nl, dtype='d')
            result += (clAB2 + 2 * clABB) * r[i_r]**2. * dr[i_r]

            ############################# DEBUG ################################
            np.savetxt('debug2/cl21_%s.dat' % run_type, result)
            ####################################################################

            print ("finished work for r=%i, dr=%.2f, avg(alpha)=%.2f, avg(beta)=%.2f, avg(result)=%.4g" % 
                (int(r[i_r]), dr[i_r], np.average(alpha[:,i_r]), 
                    np.average(beta[:,i_r]), np.average(result)))

            ############################# DEBUG ################################
            #if i_r == 0:
            #    print "finished debug -- goodbye!"
            #    exit()
            ####################################################################
            
            o_comm.Send([result,MPI.DOUBLE], dest=0, tag=1)

    f_t8 = time.time()

    if (i_rank == 0):
        print ""
        print ("Saving power spectrum to %s (not mll corrected)" 
            % fn_cl21_no_mll)

        np.savetxt(fn_cl21_no_mll, cl21)

        print ""
        print "Saving power spectrum to %s (mll corrected)" % fn_cl21
        
        cl21 = np.dot(mll_inv, cl21)
        np.savetxt(fn_cl21, cl21)

    return

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()