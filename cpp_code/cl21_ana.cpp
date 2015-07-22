/*
cl21_ana.cpp

Created on May 29, 2014
Updated on June 3, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate Cl^(2,1)_(analytical) based off of Eqn. 60 given in 
            (arXiv: 1004.1409v2, "CMB Constraints on Primordial NG...")

@inputs:    Load alpha and beta arrays from a pre-computed file (currently, 
            "l_r_alpha_beta.txt")

            na_alpha: Calculated by compute_alphabeta.f90 in 
                /fnl_Planck/alphabeta_mod, following Eqn. 49
            na_beta: Similar to na_alpha
            na_r: Similar to na_alpha
            na_dr: Similar to na_alpha

            na_cltt: power spectrum obtained from Planck maps using cltt.py
            na_ell: Similar to na_cltt

@outputs:   Analytical full skewness power spectrum,

            na_cl21

            saved to output/cl_21_ana_[i_num_r_trunc]_rsteps_[i_num_ell_trunc]_
                ellsteps.dat

@command:   Compile with:

            /home/jobryan/bin/mpic++ cl21_ana.cpp -o cl21_ana -std=c++11

            To run for a given number of ell steps, e.g., 100 steps,

            ./cl21_ana 100

            The default number of steps is 80 which will occur upon running

            ./cl21_ana

            To add the number of r steps, add an additional argument, e.g.,
            100 ell steps, 80 r steps,

            ./cl21_ana 100 80

            Currently set to not truncate the r steps; can be turned on for 
            linearly shorter run times (e.g., i_num_r_trunc = 40, etc.).

            To run on multiple processors (e.g., 4 cores),

            /home/jobryan/bin/mpirun -np 4 ./cl21_ana 100 80

            A decent intermediate run:

            /home/jobryan/bin/mpirun -np 12 ./cl21_ana 150 437

            Full run:
            
            /home/jobryan/bin/mpirun -np 12 ./cl21_ana 1499 437

*/

// C++ imports
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <math.h>
#include <tuple>
#include <regex>

// 3rd party imports
#include <mpi.h>

using namespace std;


/* globals */
int numnodes,myid,mpi_err;
#define mpi_root 0
#define _USE_MATH_DEFINES // for M_PI = 3.1415...
#define WORKTAG 1 // for MPI_Status to start
#define DIETAG 2 // for MPI_Status to end
/* end globals */

/* Natural log of factorial using Stirling's approximation -- used by w3j */

vector<double> lnfa(int i_max){

    vector<double> vd_lnfa; //natural log of factorial
    vd_lnfa.push_back(0.0);
    vd_lnfa.push_back(0.0);

    for (int i=2; i<10000; i++){

        vd_lnfa.push_back(log(i) + vd_lnfa[i-1]);

    }

        return vd_lnfa;

}

/* Calculate Wigner-3j with all m = 0 */

double w3j(int i_l1, int i_l2, int i_l3, const vector<double> &vd_lnfa){
    /* 
        Calculates the wigner-3j symbol. Please see eq. 15 of 
        http://mathworld.wolfram.com/Wigner3j-Symbol.html for more
        details. 
    */

    int i_L, i_z1, i_z2, i_z3, i_z4, i_z5, i_z6, i_z7, i_z8;
    i_L = i_l1 + i_l2 + i_l3;

    i_z1 = i_L - 2*i_l1;
    i_z2 = i_L - 2*i_l2;
    i_z3 = i_L - 2*i_l3;
    i_z4 = i_L/2;
    i_z5 = i_z4 - i_l1;
    i_z6 = i_z4 - i_l2;
    i_z7 = i_z4 - i_l3;
    i_z8 = i_L+1;

    return pow(-1,i_L/2)*exp(0.5*(vd_lnfa[i_z1] + vd_lnfa[i_z2]  
               + vd_lnfa[i_z3] - vd_lnfa[i_z8] + 2.0*(vd_lnfa[i_z4] 
               - vd_lnfa[i_z5] - vd_lnfa[i_z6] - vd_lnfa[i_z7])));
}

/* Read 1D vectors */

vector<double> read_vector(string s_fn){

    fstream o_ifs(s_fn.c_str());

    string s_line;

    vector<double> vd_data;

    while(getline(o_ifs, s_line)){

        vd_data.push_back(atof(s_line.c_str()));
    }

    //cout << "vd_data size:" << vd_data.size() << endl;

    return vd_data;
}

/* Split string on delimiter */

vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

/* Read 2D vectors */

vector< vector<double> > read_matrix(string s_fn){

    fstream o_ifs(s_fn.c_str());

    string s_line;

    vector< vector<double> > vd_data;

    while(getline(o_ifs, s_line)){

        vector<double> vd_line;
        vector<string> vs_split = split(s_line, ' ');

        for (int i=0; i<vs_split.size(); i++){

            vd_line.push_back(atof(vs_split[i].c_str()));

        }

        vd_data.push_back(vd_line);

    }

    // cout << "vd_data size:" << vd_data.size() << endl;

    return vd_data;
}

/* Write vector */

int write_vector(vector<double> vd_in, string s_fn){

    ofstream o_file;
    o_file.open(s_fn);

    int i_size1 = vd_in.size();

    cout << "Writing vector of length: " << i_size1 << endl;

    for (int i=0; i<i_size1; i++){

        ostringstream o_strs;
        o_strs << vd_in[i];
        string s_str = o_strs.str();
        o_file << s_str << endl;

    }

    o_file.close();

    return 0;

}


/* Write 3D vectors */

int write_3dvector(vector<vector<vector<double> > > vvvd_in, string s_fn){

    ofstream o_file;
    o_file.open(s_fn);

    int i_size1 = vvvd_in.size();
    int i_size2 = vvvd_in[0].size();
    int i_size3 = vvvd_in[0][0].size();

    cout << "Writing file with shape: (" << i_size1 << "," << i_size2 << "," 
         << i_size3 << ")" << endl;

    for (int i=0; i<i_size1; i++){

        for (int j=0; j<i_size2; j++){

            for (int k=0; k<i_size3; k++){

                ostringstream o_strs;
                o_strs << vvvd_in[i][j][k];
                string s_str = o_strs.str();
                o_file << s_str << " ";

            }

            o_file << endl;

        }

    }

    o_file.close();

    return 0;

}

/* Create 3d vector instantiated with zeros */

vector<vector<vector<double> > > test_cube(int i_size){

    vector<vector<vector<double> > > vvvd_return(i_size, 
        vector<vector<double> >(i_size, vector<double>(i_size)));

    return vvvd_return;

}

/*
Create a linearly or logarithmically spaced vector
*/

vector<double> spaced_vector(double d_start, double d_stop, int i_size, 
                             string s_space="lin", double d_base=10){

    vector<double> vd_return;

    double d_log_min, d_log_max, d_factor, d_delta, d_acc_delta, d_val;

    if (s_space.compare("log") == 0){

        d_log_min = log10(d_start);
        d_log_max = log10(d_stop);

        d_delta = ((double)(d_log_max - d_log_min) / (double)(i_size - 1) );
        d_acc_delta = 0;

    }
    else {

        d_factor = ((double)(d_stop - d_start) / (double)(i_size - 1) );

    }

    for (int i=0; i<i_size; i++){

        if (s_space.compare("log") == 0){

            d_val = pow(d_base, d_log_min + d_acc_delta);
            d_acc_delta += d_delta;
            vd_return.push_back(d_val);

        }
        else {

            d_val = i*d_factor + d_start;
            vd_return.push_back(d_val);

        }
            
    }

    return vd_return;

}

/*
Find closest set of points in source array to points in target array
*/

tuple< vector<double>, vector<int> > closest_points(vector<double> vd_source, 
                        vector<double> vd_target){

    vector<int> vi_ind;
    vector<double> vd_results;

    for (int i=0; i<vd_target.size(); i++){

        double d_target_val = vd_target[i];
        double d_min_dist = fabs(vd_source[0] - d_target_val);
        int i_min_index = 0;

        for (int j=0; j<vd_source.size(); j++){

            if (fabs(vd_source[j] - d_target_val) < d_min_dist){

                d_min_dist = fabs(vd_source[j] - d_target_val);
                i_min_index = j;

            }

        }

        vi_ind.push_back(i_min_index);
        vd_results.push_back(vd_source[i_min_index]);
    }

    return tuple< vector<double>, vector<int> >(vd_results, vi_ind);

}

/*
Select subarrays given number of points, maximum point in selection, and spacing
*/

tuple< vector<double>, vector<int> > sub_arr(vector<double> vd_arr, 
                        int i_num_pts, double d_max, string s_spacing="lin"){
    
    // Select sub array of array using number of points, max, and spacing type

    vector<double> vd_target_spacing;

    if (s_spacing.compare("log") == 0){
    
        vd_target_spacing = spaced_vector(
            *min_element(vd_arr.begin(), vd_arr.end()), d_max, 
            i_num_pts, "log");

    }
    else{
    
        vd_target_spacing = spaced_vector(
            *min_element(vd_arr.begin(), vd_arr.end()), d_max, 
            i_num_pts, "lin");

    }

    vector<double> vd_sub;
    vector<int> vi_ind;

    tie(vd_sub, vi_ind) = closest_points(vd_arr, vd_target_spacing);

    return tuple< vector<double>, vector<int> > (vd_sub, vi_ind);

}

/*
Select subarray given a vector of indices
*/

template<class T>
vector<T> sub_arr(vector<T> vt_arr, vector<int> vi_ind){

    vector<T> vt_return;

    for (int i=0; i<vi_ind.size(); i++){

        vt_return.push_back(vt_arr[vi_ind[i]]);

    }

    return vt_return;

}

/*
Slice array
*/

template<class T>
vector< vector<T> > slice_arr(vector< vector<T> > vvt_arr, vector<int> vi_ind1,
                              vector<int> vi_ind2){

    vector< vector<T> > vvt_return;

    int i_ind1, i_ind2;

    for (int i=0; i<vi_ind1.size(); i++){

        i_ind1 = vi_ind1[i];
        vector<T> vt_row;

        for (int j=0; j<vi_ind2.size(); j++){

            i_ind2 = vi_ind2[j];

            vector<T> vt_tmp = vvt_arr[i_ind1];

            vt_row.push_back(vt_tmp[i_ind2]);

            //cout << "selecting row " << i_ind1 << " and column " << i_ind2 << endl;
            //cout << "value: " << vt_tmp[i_ind2] << endl;

        }

        vvt_return.push_back(vt_row);

    }

    return vvt_return;

}

/*
Cartesian product of three vectors
*/

template<class T>
vector<vector<T> > cart_triple_prod(vector<T> vt_1, vector<T> vt_2, 
                                    vector<T> vt_3){

    vector<vector<T> > vvt_tuples;
    int i_size1 = vt_1.size();
    int i_size2 = vt_2.size();
    int i_size3 = vt_3.size();

    for (int i=0; i<i_size1; i++){

        for (int j=0; j<i_size2; j++){

            for (int k=0; k<i_size3; k++){

                T at_entry[] = {vt_3[k], vt_2[j], vt_1[i]};
                vector<T> vt_entry (at_entry, 
                    at_entry + sizeof(at_entry) / sizeof(at_entry[0]) );

                vvt_tuples.push_back(vt_entry);

            }

        }

    }

    return vvt_tuples;

}

/* Return index for cartesian product */

vector<int> cart_index(int i_index, vector<int> vi_dims){
    
    /*
    Returns the index from a cartesian product of vectors which each range from
    1 to n_i (where n_i is the ith entry of vi_dims)
    */

    reverse(vi_dims.begin(), vi_dims.end());

    vector<int> vi_tuple(vi_dims.size(), 0);

    // Error checking

/*    int i_max_size = 1;

    for (int i=0; i<vi_dims.size(); i++){

        i_max_size *= vi_dims[i];

    }

    if (i_index > i_max_size){

        cout << "ERROR: Index too large for cart_index!" << endl;

        return vi_tuple;

    }*/

    vector<int> vi_divisor(vi_dims.size(), 1);

    for (int i=vi_divisor.size()-1; i>-1; i--){

        if (i < vi_divisor.size() - 1 ){

            vi_divisor[i] = vi_dims[i] * vi_divisor[i+1];

        }

    }

    int i_quotient = i_index;

    for (int i=0; i<vi_dims.size(); i++){

        vi_tuple[i] = (i_quotient / vi_divisor[i]);
        i_quotient = i_quotient % vi_divisor[i];

    }

    return vi_tuple;

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

/*
Main: Default run
*/

int main(int i_argc, char* s_argv[]){

    // MPI Initialization
    
    int i_rank, i_size;

    MPI_Init(&i_argc, &s_argv);              /* starts MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);  /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &i_size);  /* get number of processes */

    // Command line argument for i_ell_trunc
    int i_ell_trunc = 9999;
    int i_r_trunc = 9999;

    if (i_argc > 1){

        stringstream str(s_argv[1]);
        str >> i_ell_trunc;

    }

    if (i_argc > 2){

        stringstream str(s_argv[2]);
        str >> i_r_trunc;

    }

    /*
    Load and plot data (ell, r, dr, alpha, beta)
    */

    // Load parameters
    if (i_rank == 0){

        cout << "Setting load parameters:" << endl;

    }

    string s_fn_alpha = "../data/alpha_ell_1499_r_437.txt";
    string s_fn_beta = "../data/beta_ell_1499_r_437.txt";
    string s_fn_ell = "../data/ell_1499.txt";
    string s_fn_r = "../data/r_437.txt";
    string s_fn_dr = "../data/dr_r_437.txt";

    int i_num_r = 437; // columns
    int i_num_ell = 1499; // rows

    if (i_rank == 0){

        cout << "loading from: " << s_fn_alpha << endl
             << "loading from: " << s_fn_beta << endl
             << "loading from: " << s_fn_ell << endl
             << "loading from: " << s_fn_r << endl
             << "loading from: " << s_fn_dr << endl
             << "r steps: " << i_num_r << ", ell steps: " << i_num_ell << endl 
             << endl;

    }

    // Load alpha, beta data
    if (i_rank == 0){
    
        cout << "Loading data..." << endl 
             << "(displaying array shapes)" << endl;
    
    }

    vector< vector<double> > vvd_alpha = read_matrix(s_fn_alpha);
    vector< vector<double> > vvd_beta = read_matrix(s_fn_beta);

    vector<double> vd_l = read_vector(s_fn_ell);
    vector<double> vd_r = read_vector(s_fn_r);
    vector<double> vd_dr = read_vector(s_fn_dr);

    // Load power spectrum
    string s_fn_clttp = "../output/na_cltt_ell_1499.txt";
    vector<double> vd_cltt = read_vector(s_fn_clttp);

    if (i_rank == 0){

        cout << "alpha: (" << vvd_alpha.size() << "," << vvd_alpha[0].size() 
             << "), beta: (" << vvd_beta.size() << "," << vvd_beta[0].size()
             << "), cltt: " << vd_cltt.size() << endl;
        cout << "l: " << vd_l.size() << " r: " << vd_r.size() << " dr: " 
             << vd_dr.size() << endl << endl;

    }

    /*
    Calculate bispectrum -- MPI optimized
    */

    // Run Parameters
    if (i_rank == 0){

        cout << "(Running calculation with " << i_size << " cores)" << endl;
        cout << "Setting run parameters for bispectrum calculation:" << endl;

    }
    
    int i_num_ell_trunc;

    if (i_ell_trunc != 9999){

        i_num_ell_trunc = i_ell_trunc;

    }
    else {

        i_num_ell_trunc = 80;

    }

    
    int i_num_r_trunc = vd_r.size();

    if (i_r_trunc != 9999){

        i_num_r_trunc = i_r_trunc;

    }
    else {

        i_num_r_trunc = 40;

    }

    if (i_rank == 0){

        cout << "r steps (trunc): " << i_num_r_trunc << " ell steps (trunc): "
             << i_num_ell_trunc << endl << endl;

    }

    // Chop down arrays for reduced bispectrum calculation

    string s_trunc_type = "log";

    if (i_rank == 0){

        cout << "Truncating arrays for bispectrum calculation:" << endl
             << "(displaying array shapes; using " << s_trunc_type 
                << " truncation)" << endl;

    }

    vector<int> vi_l_ind, vi_cltt_ind, vi_r_ind;

    tie(vd_l, vi_l_ind) = sub_arr(vd_l, i_num_ell_trunc, 
                   *max_element(vd_l.begin(), vd_l.end()), s_trunc_type);

    vd_cltt = sub_arr(vd_cltt, vi_l_ind);

    tie(vd_r, vi_r_ind) = sub_arr(vd_r, i_num_r_trunc, 
                   *max_element(vd_r.begin(), vd_r.end()), "lin");

    vd_dr = sub_arr(vd_dr, vi_r_ind);

    // Slicing vectors in C++...a little kludgy at the moment...

    vector<double> vd_l_ind_all = spaced_vector(0., (double)vvd_alpha.size()-1., 
                                                vvd_alpha.size());
    vector<int> vi_l_ind_all(vd_l_ind_all.begin(), vd_l_ind_all.end());

    vector<vector<double> > vvd_alpha_tmp = slice_arr(vvd_alpha, vi_l_ind_all, 
                                                      vi_r_ind);
    vector<vector<double> > vvd_beta_tmp = slice_arr(vvd_beta, vi_l_ind_all, 
                                                     vi_r_ind);

    vector<double> vd_r_ind_all = spaced_vector(0., (double)vi_r_ind.size()-1., 
                                                vi_r_ind.size());
    vector<int> vi_r_ind_all(vd_r_ind_all.begin(), vd_r_ind_all.end());

    vvd_alpha = slice_arr(vvd_alpha_tmp, vi_l_ind, vi_r_ind_all);
    vvd_beta = slice_arr(vvd_beta_tmp, vi_l_ind, vi_r_ind_all);

    if (i_rank == 0){

        cout << "alpha: (" << vvd_alpha.size() << "," << vvd_alpha[0].size() 
             << "), beta: (" << vvd_beta.size() << "," << vvd_beta[0].size()
             << ")" << endl
             << "l: " << vd_l.size()  << ", cltt: " << vd_cltt.size() 
             << ", r: " << vd_r.size() << ", dr: " << vd_dr.size() 
             << endl << endl;

    }


    // Calculate bispectrum
    if (i_rank == 0){
    
        cout << "Calculating analytical reduced bispectrum..." << endl;

    }

    double d_fnl = 1.0;
    
    clock_t o_start;

    if (i_rank == 0){

        o_start = clock();
    
    }


    /*
    Calculate bispectrum
    */

    // b_red[l1,l2,l3] = 2 fnl int[ dr r^2 [a(r,l1) b(r,l2) b(r,l3) + cyc. perm] ]
    // B[l1,l2,l3] = sqrt( (2*l1+1)*(2*l1+1)*(2*l1+1)/(4*M_PI) ) * w3j(l1,l2,l3) * b_red[l1,l2,l3]

    int i_num_l = vd_l.size();
    i_num_r = vd_r.size();

    vector<double> vd_cl21(i_num_l, 0);

    int ai_dims[] = {i_num_l, i_num_l, i_num_l};
    vector<int> vi_dims (ai_dims, 
        ai_dims + sizeof(ai_dims) / sizeof(ai_dims[0]) );

    //master loop

    if (i_rank == 0) {

        // declarations for work input / result output

        int i_l1ind, i_l2ind, i_l3ind; 
        double d_l1, d_l2, d_l3, d_r_sum;
        vector<double> vd_result(7,0); // il1, il2, il3, dl1, dl2, dl3, d_r_sum
        vector<int> vi_work(3,0); // il1, il2, il3

        MPI_Status o_status;        

        cout << "Sending out initial jobs to all cores..." << endl;

        for (int i_rank_out = 1; i_rank_out < i_size; i_rank_out++) {

            vi_work = cart_index(i_rank_out-1, vi_dims);

/*            cout << "sending to core " << i_rank_out << endl;
            cout << "job " << vi_work[0] << "," << vi_work[1] << "," << vi_work[2] << endl;
*/
            /* Send job to each rank */

            MPI_Send(&vi_work[0],          /* message buffer */
                     vi_work.size(),    /* one data item */
                     MPI_INT,           /* data item is an integer */
                     i_rank_out,        /* destination process rank */
                     WORKTAG,           /* user chosen message tag */
                     MPI_COMM_WORLD);   /* default communicator */

        }

        /* Loop over getting new work requests until there is no more work
           to be done */

        vi_work = cart_index(i_size-1, vi_dims);
        int i_l1_start = vi_work[0];
        int i_l2_start = vi_work[1];
        int i_l3_start = vi_work[2];

        for (int i_l1=i_l1_start; i_l1<i_num_l; i_l1++) {

                    if (i_l1 % (i_num_l / 10) == 0){

                        cout << "Finished " << (i_l1 * 100 / i_num_l)
                             << "% of jobs..." << endl;

                    }

            for (int i_l2=i_l2_start; i_l2<i_num_l; i_l2++) {

                for (int i_l3=i_l3_start; i_l3<i_num_l; i_l3++) {

                    vi_work[0] = i_l1;
                    vi_work[1] = i_l2;
                    vi_work[2] = i_l3;

                    /* Receive results from a slave */

                    MPI_Recv(&vd_result[0],     /* message buffer */
                             vd_result.size(),  /* one data item */
                             MPI_DOUBLE,        /* of type double real */
                             MPI_ANY_SOURCE,    /* receive from any sender */
                             MPI_ANY_TAG,       /* any type of message */
                             MPI_COMM_WORLD,    /* default communicator */
                             &o_status);        /* info about the received message */

                    /*cout << "sending job " << vi_work[0] << "," << vi_work[1] 
                         << "," << vi_work[2] << " to core " 
                         << o_status.MPI_SOURCE << endl;*/

                    /* Send the slave a new work unit */

                    MPI_Send(&vi_work[0],              /* message buffer */
                             vi_work.size(),        /* one data item */
                             MPI_INT,               /* data item is an integer */
                             o_status.MPI_SOURCE,   /* to who we just received from */
                             WORKTAG,               /* user chosen message tag */
                             MPI_COMM_WORLD);       /* default communicator */

                    /* Process received job */

                    i_l1ind = (int) vd_result[0];
                    i_l2ind = (int) vd_result[1];
                    i_l3ind = (int) vd_result[2];

                    d_l1 = vd_result[3];
                    d_l2 = vd_result[4];
                    d_l3 = vd_result[5];

                    d_r_sum = vd_result[6];

                    //cout << "receiving sum " << d_r_sum << endl;

                    vd_cl21[i_l1ind] += d_r_sum;

                }

            }

        }

        for (int i_rank_out = 1; i_rank_out < i_size; i_rank_out++) {


            /* Receive results from a slave */

            MPI_Recv(&vd_result[0],     /* message buffer */
                     vd_result.size(),  /* one data item */
                     MPI_DOUBLE,        /* of type double real */
                     MPI_ANY_SOURCE,    /* receive from any sender */
                     MPI_ANY_TAG,       /* any type of message */
                     MPI_COMM_WORLD,    /* default communicator */
                     &o_status);        /* info about the received message */

            /* Process received job */

            i_l1ind = (int) vd_result[0];
            i_l2ind = (int) vd_result[1];
            i_l3ind = (int) vd_result[2];

            d_l1 = vd_result[3];
            d_l2 = vd_result[4];
            d_l3 = vd_result[5];

            d_r_sum = vd_result[6];

            vd_cl21[i_l1ind] += d_r_sum;

            /* Tell the slave to exit */

            int i_dead = 0;

            MPI_Send(&i_dead, 
                     1, 
                     MPI_INT, 
                     o_status.MPI_SOURCE, 
                     DIETAG, 
                     MPI_COMM_WORLD);
        }

    } 

    //slave loop

    else {

        vector<int> vi_work(3,0); // il1, il2, il3
        double d_r_sum = 0.0;

        MPI_Status o_status;

        vector<double> vd_lnfa = lnfa(10000);

        while (1) {

            /* Receive a message from the master */
            
            MPI_Recv(&vi_work[0], vi_work.size(), MPI_INT, 0, MPI_ANY_TAG, 
                MPI_COMM_WORLD, &o_status);

            //cout << "received job " << vi_work[0] << "," << vi_work[1] << "," << vi_work[2] << endl;

            /* Check the tag of the received message */

            if (o_status.MPI_TAG == DIETAG){

                break;

            }

            /* Do the work */

            d_r_sum = 0.0;

            int i_l1ind = vi_work[0];
            int i_l2ind = vi_work[1];
            int i_l3ind = vi_work[2];

            double d_l1 = vd_l[i_l1ind];
            double d_l2 = vd_l[i_l2ind];
            double d_l3 = vd_l[i_l3ind];

            if ((fabs(d_l1 - d_l2) <= d_l3) && (fabs(d_l1 + d_l2) >= d_l3)){

                for (int i_r=0; i_r<i_num_r; i_r++){

                    d_r_sum += (2.0 * d_fnl * 
                        vd_dr[i_r] * pow(vd_r[i_r],2) * 
                        (vvd_alpha[i_l1ind][i_r] * vvd_beta[i_l2ind][i_r] 
                            * vvd_beta[i_l3ind][i_r]
                        + vvd_alpha[i_l3ind][i_r] * vvd_beta[i_l1ind][i_r] 
                        * vvd_beta[i_l2ind][i_r]
                        + vvd_alpha[i_l2ind][i_r] * vvd_beta[i_l3ind][i_r] 
                        * vvd_beta[i_l1ind][i_r]));

                }


                d_r_sum *= sqrt( (2*d_l1 + 1) * (2*d_l2 + 1) * (2*d_l3 + 1) 
                        / (4 * M_PI))
                        * w3j((int) d_l1, (int) d_l2, (int) d_l3, vd_lnfa);

                d_r_sum = pow(d_r_sum,2) / vd_cltt[i_l1ind] / vd_cltt[i_l2ind] /
                            vd_cltt[i_l3ind] / (2*d_l1 + 1.);

            }

            else {

                d_r_sum = 0.0;

            }

            double ad_data[] = {(double)i_l1ind, (double)i_l2ind, 
                        (double)i_l3ind, d_l1, d_l2, d_l3, d_r_sum};
            vector<double> vd_result (ad_data, 
                    ad_data + sizeof(ad_data) / sizeof(ad_data[0]) );

            MPI_Send(&vd_result[0], vd_result.size(), MPI_DOUBLE, 0, 1, 
                MPI_COMM_WORLD);

            //cout << "finished job " << vi_work[0] << "," << vi_work[1] << "," << vi_work[2] << " with value " << d_r_sum << endl;

        }

    }

    MPI_Finalize();

    if (i_rank == 0){

        cout << endl << "Time for skewness power spectrum calculation: " 
             << (clock() - o_start) / (double)(CLOCKS_PER_SEC / 1000) 
             << " ms" << endl;

    }

    // Save bispectrum as numpy array

    ostringstream o_strs;
    o_strs << "../output/cl_21_ana_" << i_num_r_trunc << "_rsteps_" 
           << i_num_ell_trunc << "_ellsteps.dat";
    string s_fn_cl21 = o_strs.str();

    ostringstream o_strs2;
    o_strs2 << "../output/ell_out_" << i_num_r_trunc << "_rsteps_" 
           << i_num_ell_trunc << "_ellsteps.dat";
    string s_fn_ell_out = o_strs2.str();

    if (i_rank == 0){

        cout << "Saving skewness power spectrum to " << s_fn_cl21 << endl;
        write_vector(vd_cl21, s_fn_cl21);
        cout << "Saving ell slices to " << s_fn_ell_out << endl;
        write_vector(vd_l, s_fn_ell_out);

    }

    if (i_rank == 0){

        cout << "Done!" << endl << endl ;

    }

    if (i_rank == 0){

        int i_print_step = i_num_l / 10;

        for (int i=0; i<i_num_l; i+=i_print_step){

                    cout << "ell: " << vd_l[i] << ", cl21_ana: " 
                         << vd_cl21[i] << endl;
        }
    }

    return 0;
}