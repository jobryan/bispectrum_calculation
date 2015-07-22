/*
kl31_ana.cpp

Created on June 24, 2014
Updated on June 24, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate Kl^(3,1) based off of Eqn. 76 given in 
            (arXiv: 1004.1409v2, "CMB Constraints on Primordial NG...")

@inputs:    Load alpha and beta arrays from a pre-computed file (currently, 
            "l_r_alpha_beta.txt")

            na_alpha: Calculated by compute_alphabeta.f90 in 
                /fnl_Planck/alphabeta, following Eqn. 49
            na_beta: Similar to na_alpha
            na_r: Similar to na_alpha
            na_dr: Similar to na_alpha

            na_cltt: power spectrum obtained from Planck maps using cltt.py
            na_ell: Similar to na_cltt

            na_clcurv: curvature perturbation power spectrum obtained using
                /fnl_Planck/alphabeta/compute_clcurve.f90

@outputs:   Analytical kurtosis estimators,

            na_kl31

            saved to 

            and output/kl_31_ana_[i_num_r_trunc]_rsteps_[i_num_L]_Lsteps_[i_lmax]
                _lmax.dat

@command:   Compile with:

            /home/jobryan/bin/mpic++ kl31_ana.cpp -o kl31_ana -std=c++11

            To run for a given number of ell steps, e.g., 100 steps,

            ./kl31_ana 100

            The default number of steps is 80 which will occur upon running

            ./kl31_ana

            To add the number of r steps, add an additional argument, e.g.,
            100 ell steps, 80 r steps,

            ./kl31_ana 100 80

            Currently set to not truncate the r steps; can be turned on for 
            linearly shorter run times (e.g., i_num_r_trunc = 40, etc.).

            To run with lmax other than i_num_ell (the total length of cltt), 
            set the third parameter:

            ./kl31_ana 100 80 300

            To run on multiple processors (e.g., 12 cores),

            /home/jobryan/bin/mpirun -np 12 ./kl31_ana 10 10 50

            If running from an incomplete run, (or if you want to start at an 
            intermediate point), in the location s_fn_cur_job 
            (../output/vi_cur_job_kl31.dat), there should be a text file with 5 lines
            with the l1, l2, l3, l4, and L starting locations on each line,
            e.g.,

            10
            20
            30
            40
            50

            If that file exists, kl31_ana will use its values as its starting
            location.

*/

// C++ imports
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
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

    for (int i=2; i<i_max; i++){

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

template<class T>
int write_vector(vector<T> vT_in, string s_fn){

    ofstream o_file;
    o_file.open(s_fn);

    int i_size1 = vT_in.size();

    cout << "Writing vector of length: " << i_size1 << endl;

    for (int i=0; i<i_size1; i++){

        ostringstream o_strs;
        o_strs << vT_in[i];
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

/* Read 3D vectors */

vector<vector<vector<double> > > read_3dvector(string s_fn, int i_size){

    fstream o_ifs(s_fn.c_str());

    string s_line;

    vector<vector<vector<double> > > vvvd_data;

    for (int i=0; i<i_size; i++){

        vector<vector<double> > vvd_block;

        for (int j=0; j<i_size; j++){

            getline(o_ifs, s_line);
            vector<double> vd_line;
            vector<string> vs_split = split(s_line, ' ');

            for (int k=0; k<i_size; k++){

                vd_line.push_back(atof(vs_split[k].c_str()));

            }

            vvd_block.push_back(vd_line);

        }

        vvvd_data.push_back(vvd_block);

    }

    return vvvd_data;

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
Slice array -- vector
*/

template<class T>
vector<T> slice_arr(vector<T> vt_arr, int i_ind1, int i_ind2){

    vector<T> vt_return;

    for (int i=i_ind2; i>=i_ind1; i--){

        vt_return.push_back(vt_arr[i]);

    }

    return vt_return;

}

/*
Slice array -- matrix
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

/*
Cartesian product of five vectors
*/

template<class T>
vector<vector<T> > cart_quintuple_prod(vector<T> vt_1, vector<T> vt_2, 
                            vector<T> vt_3, vector<T> vt_4, vector<T> vt_5){

    vector<vector<T> > vvt_tuples;
    int i_size1 = vt_1.size();
    int i_size2 = vt_2.size();
    int i_size3 = vt_3.size();
    int i_size4 = vt_4.size();
    int i_size5 = vt_5.size();

    for (int i=0; i<i_size1; i++){

        for (int j=0; j<i_size2; j++){

            for (int k=0; k<i_size3; k++){

                for (int l=0; l<i_size4; l++){

                    for (int m=0; m<i_size5; m++){

                        T at_entry[] = {vt_5[m], vt_4[l], vt_3[k], vt_2[j], 
                                        vt_1[i]};
                        vector<T> vt_entry (at_entry, 
                            at_entry + sizeof(at_entry) / sizeof(at_entry[0]) );

                        vvt_tuples.push_back(vt_entry);

                    }

                }

            }

        }

    }

    return vvt_tuples;

}

vector<int> cart_index(int i_index, vector<int> vi_dims){
    
    /*
    Returns the index from a cartesian product of vectors which each range from
    1 to n_i (where n_i is the ith entry of vi_dims)
    */

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

    int i_dims = vi_dims.size();

    for (int i=0; i<i_dims; i++){

        if (i == i_dims - 1){  

            vi_tuple[i] = i_index % vi_dims[i];

        }
        else{

            vi_tuple[i] = i_index;

            for (int j=i+1; j<i_dims; j++){

                vi_tuple[i] /= vi_dims[j];

            }

            vi_tuple[i] %= vi_dims[i];

        }

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
    
    int i_rank, i_size, i_continuing_job, i_l1_start, i_l2_start, 
        i_l3_start, i_l4_start, i_L_start;

    MPI_Init(&i_argc, &s_argv);              /* starts MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);  /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &i_size);  /* get number of processes */

    // Command line argument for i_ell_trunc
    int i_ell_trunc = 9999;
    int i_r_trunc = 9999;
    int i_lmax = 9999;

    if (i_argc > 1){

        stringstream str(s_argv[1]);
        str >> i_ell_trunc;

    }

    if (i_argc > 2){

        stringstream str(s_argv[2]);
        str >> i_r_trunc;

    }

    string s_fn_cur_job = "../output/vi_cur_job_kl31.dat";
    ifstream o_cur_job(s_fn_cur_job, ios::binary);

    if (i_argc > 3){

        stringstream str(s_argv[3]);
        str >> i_lmax;

    }

    if (o_cur_job){

        vector<double> vd_previous_job = read_vector(s_fn_cur_job);
        i_l1_start = (int)vd_previous_job[0];
        i_l2_start = (int)vd_previous_job[1];
        i_l3_start = (int)vd_previous_job[2];
        i_l4_start = (int)vd_previous_job[3];
        i_L_start = (int)vd_previous_job[4];

        i_continuing_job = 1;

    }
    else {

        i_l1_start = 0;
        i_l2_start = 0;
        i_l3_start = 0;
        i_l4_start = 0;
        i_L_start = 0;

        i_continuing_job = 0;

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
    string s_fn_clcurv = "../output/na_clcurv_ell_1499.txt";
    vector<double> vd_clcurv = read_vector(s_fn_clcurv);

    if (i_rank == 0){

        cout << "alpha: (" << vvd_alpha.size() << "," << vvd_alpha[0].size() 
             << "), beta: (" << vvd_beta.size() << "," << vvd_beta[0].size()
             << "), cltt: " << vd_cltt.size()  << ", clcurv: " 
             << vd_clcurv.size() << endl;
        cout << "l: " << vd_l.size() << " r: " << vd_r.size() << " dr: " 
             << vd_dr.size() << endl << endl;

    }

    /*
    Calculate trispectrum -- MPI optimized
    */

    // Run Parameters
    if (i_rank == 0){

        cout << "(Running calculation with " << i_size << " cores)" << endl;
        cout << "Setting run parameters for trispectrum calculation:" << endl;

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

    if (i_lmax == 9999){

        i_lmax = i_num_ell;

    }

    if (i_rank == 0){

        cout << "r steps (trunc): " << i_num_r_trunc << " ell steps (trunc): "
             << i_num_ell_trunc << endl << endl;

    }

    // Chop down arrays for reduced trispectrum calculation -- keep original
    // vd_l for later use (all l_i use original; L only uses truncation)

    string s_trunc_type = "log";

    if (i_rank == 0){

        cout << "Truncating arrays for trispectrum calculation:" << endl
             << "(displaying array shapes; using " << s_trunc_type 
                << " truncation)" << endl;

    }

    vector<double> vd_L;
    vector<int> vi_l_ind, vi_L_ind, vi_cltt_ind, vi_clcurv_ind, vi_r_ind;

    tie(vd_L, vi_L_ind) = sub_arr(vd_l, i_num_ell_trunc, (double)i_lmax, 
                                    s_trunc_type);

    tie(vd_r, vi_r_ind) = sub_arr(vd_r, i_num_r_trunc, 
                   *max_element(vd_r.begin(), vd_r.end()), "lin");

    vd_dr = sub_arr(vd_dr, vi_r_ind);

    // Slicing vectors in C++...a little kludgy at the moment...

    vector<double> vd_l_ind_all = spaced_vector(0., (double)vvd_alpha.size()-1., 
                                                vvd_alpha.size());
    vector<int> vi_l_ind_all(vd_l_ind_all.begin(), vd_l_ind_all.end());

    vvd_alpha = slice_arr(vvd_alpha, vi_l_ind_all, vi_r_ind);
    vvd_beta = slice_arr(vvd_beta, vi_l_ind_all, vi_r_ind);

    if (i_rank == 0){

        cout << "alpha: (" << vvd_alpha.size() << "," << vvd_alpha[0].size() 
             << "), beta: (" << vvd_beta.size() << "," << vvd_beta[0].size()
             << ")" << endl
             << "l_i: " << i_lmax  << ", L: " << vd_L.size()
             << ", cltt: " << vd_cltt.size() << ", clcurv: " << vd_clcurv.size() 
             << ", r: " << vd_r.size() << ", dr: " << vd_dr.size() 
             << endl << endl;

    }

    // Calculate bispectrum
    if (i_rank == 0){
    
        cout << "Calculating kl31..." << endl;

    }

    double d_tnl = 1.0;
    double d_gnl = 1.0;
    
    clock_t o_start;
    double d_time_elapsed;
    double d_time_stop = 150.9;
    //double d_time_stop = 15.9;
    //double d_time_stop = 0.010;
    int i_early_write = 0;

    if (i_rank == 0){

        o_start = clock();
        d_time_elapsed = 0.0;
    
    }

    // Filenames for kl31 and ell outputs

    ostringstream o_strs;
    o_strs << "../output/kl_31_ana_" << i_num_r_trunc << "_rsteps_" 
           << i_num_ell_trunc << "_Lsteps_" << i_lmax  << "_lmax_"
           << (int)d_gnl << "_gnl_" << (int)d_tnl << "_tnl.dat";
    string s_fn_kl31 = o_strs.str();

    ostringstream o_strs3;
    o_strs3 << "../output/ell_out_" << i_num_r_trunc << "_rsteps_" 
           << i_num_ell_trunc << "_Lsteps_" << i_lmax << "_lmax_"
           << (int)d_gnl << "_gnl_" << (int)d_tnl << "_tnl_kl31.dat";
    string s_fn_ell_out = o_strs3.str();

    /*
    Calculate trispectrum
    */

/*    T[l1,l2,l3,l4,L] ~= ( tnl * cl_curv[L] * (cl_tt[l1] + cl_tt[l2]) 
                                           * (cl_tt[l3] + cl_tt[l4])
                    + 2 gnl * int r^2 dr beta[l2] beta[l4] 
                        * (alpha[l1][r] beta[l3][r] + alpha[l3][r] beta[l1][r]))
                    * sqrt( (2l1+1)(2l2+1)(2L+1) / )
*/

/*    int i_all_counter = 0;
    int i_sum_counter = 0;*/

    int i_num_l = vd_l.size();
    int i_num_L = vd_L.size();
    i_num_r = vd_r.size();

    vector<double> vd_kl31(i_num_L, 0);

    int ai_dims[] = {i_lmax, i_lmax, i_lmax, i_lmax, i_num_L};
    vector<int> vi_dims (ai_dims, 
        ai_dims + sizeof(ai_dims) / sizeof(ai_dims[0]) );

    //master loop

    if (i_rank == 0) {

        // declarations for work input / result output

        if (i_continuing_job > 0){

            cout << "Continuing job from " << i_l1_start << "," << i_l2_start 
                 << "," << i_l3_start << "," << i_l4_start << "," << i_L_start
                 << endl;

           vd_kl31 = read_vector(s_fn_kl31);

        }

        int i_l1ind, i_l2ind, i_l3ind, i_l4ind, i_Lind; 
        double d_l1, d_l2, d_l3, d_l4, d_L, d_r_sum_31;
        vector<double> vd_result(11,0); // il1, il2, il3, il4, iL, dl1, dl2, dl3, dl4, dL, d_r_sum_31
        vector<int> vi_work(5,0); // il1, il2, il3, il4, iL

        MPI_Status o_status;

        int i_dead = 0;

        cout << "Sending out initial jobs to all cores..." << endl;

        for (int i_rank_out = 1; i_rank_out < i_size; i_rank_out++) {

            if (i_continuing_job > 0){

                int ai_job[] = {
                    i_l4_start + ((i_l4_start + i_rank_out-1) / i_num_L / i_num_l / i_num_l / i_num_l) % i_num_l,
                    i_l1_start + ((i_l1_start + i_rank_out-1) / i_num_L / i_num_l / i_num_l) % i_num_l,
                    i_l2_start + ((i_l2_start + i_rank_out-1) / i_num_L / i_num_l) % i_num_l,
                    i_l3_start + ((i_l3_start + i_rank_out-1) / i_num_L) % i_num_l, 
                    (i_L_start + i_rank_out-1) % i_num_L};
                vi_work.assign(ai_job, 
                    ai_job + sizeof(ai_job) / sizeof(ai_job[0]));

            }
            else{

                vi_work = cart_index(i_rank_out-1, vi_dims);

            }

            /* Send it to each rank */

            MPI_Send(&vi_work[0],       /* message buffer */
                     vi_work.size(),    /* one data item */
                     MPI_INT,           /* data item is an integer */
                     i_rank_out,        /* destination process rank */
                     WORKTAG,           /* user chosen message tag */
                     MPI_COMM_WORLD);   /* default communicator */

        }

        /* Loop over getting new work requests until there is no more work
           to be done */

        if (i_continuing_job > 0){

            int ai_next_job[] = {
                    i_l4_start + ((i_l4_start + i_size-1) / i_num_L / i_num_l / i_num_l / i_num_l) % i_num_l,
                    i_l1_start + ((i_l1_start + i_size-1) / i_num_L / i_num_l / i_num_l) % i_num_l,
                    i_l2_start + ((i_l2_start + i_size-1) / i_num_L / i_num_l) % i_num_l,
                    i_l3_start + ((i_l3_start + i_size-1) / i_num_L) % i_num_l, 
                    (i_L_start + i_size-1) % i_num_L};
            vi_work.assign(ai_next_job, 
                ai_next_job + sizeof(ai_next_job) / sizeof(ai_next_job[0]));

        }
        else{

            vi_work = cart_index(i_size-1, vi_dims);

        }

        i_l4_start = vi_work[0];
        i_l1_start = vi_work[1];
        i_l2_start = vi_work[2];
        i_l3_start = vi_work[3];
        i_L_start = vi_work[4];
        
/*        cout << "i_l1_start: " << i_l1_start << " i_l2_start: " << i_l2_start 
             << " i_l3_start: " << i_l3_start << " i_l4_start: " << i_l4_start
             << " i_L_start: " << i_L_start << endl;*/

        for (int i_l4=i_l4_start; i_l4<i_lmax; i_l4++){

            if (i_l4 % (i_lmax / 10) == 0){

            cout << "Finished " << (i_l4 * 100 / i_lmax)
                 << "% of jobs..." << endl;

            }

            for (int i_l1=i_l1_start; i_l1<i_lmax; i_l1++){

                for (int i_l2=i_l2_start; i_l2<i_lmax; i_l2++){

                    d_l1 = vd_l[i_l1];
                    d_l2 = vd_l[i_l2];
                    d_l4 = vd_l[i_l4];

                    if ((fabs(d_l1 - d_l2) <= d_l4) && (fabs(d_l1 + d_l2) >= d_l4)){

                        for (int i_l3=i_l3_start; i_l3<i_lmax; i_l3++){

                            for (int i_L=i_L_start; i_L<i_num_L; i_L++){

                                d_l3 = vd_l[i_l3];
                                d_L = vd_L[i_L];

                                if ((fabs(d_l3 - d_L) <= d_l4) && (fabs(d_l3 + d_L) >= d_l4)){

                                    vi_work[0] = i_l4;
                                    vi_work[1] = i_l1;
                                    vi_work[2] = i_l2;
                                    vi_work[3] = i_l3;
                                    vi_work[4] = i_L;

                                    /* Receive results from a slave */

                                    MPI_Recv(&vd_result[0],     /* message buffer */
                                             vd_result.size(),  /* one data item */
                                             MPI_DOUBLE,        /* of type double real */
                                             MPI_ANY_SOURCE,    /* receive from any sender */
                                             MPI_ANY_TAG,       /* any type of message */
                                             MPI_COMM_WORLD,    /* default communicator */
                                             &o_status);        /* info about the received message */

                                    /* Send the slave a new work unit */

                                    MPI_Send(&vi_work[0],           /* message buffer */
                                             vi_work.size(),        /* one data item */
                                             MPI_INT,               /* data item is an integer */
                                             o_status.MPI_SOURCE,   /* to who we just received from */
                                             WORKTAG,               /* user chosen message tag */
                                             MPI_COMM_WORLD);       /* default communicator */

                                    /* Process received job */

                                    i_l1ind = (int) vd_result[0];
                                    i_l2ind = (int) vd_result[1];
                                    i_l3ind = (int) vd_result[2];
                                    i_l4ind = (int) vd_result[3];
                                    i_Lind = (int) vd_result[4];

                                    d_l1 = vd_result[5];
                                    d_l2 = vd_result[6];
                                    d_l3 = vd_result[7];
                                    d_l4 = vd_result[8];
                                    d_L = vd_result[9];

                                    d_r_sum_31 = vd_result[10];
                                    vd_kl31[i_Lind] += d_r_sum_31;

/*                                    i_all_counter++;

                                    if (d_r_sum_31 > 0){

                                        i_sum_counter++;

                                    }
*/
                                    /* Check timer: if time > 15.5 hrs, write kl31 to file;
                                        also write progress (i_l1,...,i_L) to file */

                                    d_time_elapsed = ((clock() - o_start) 
                                        / (double)(CLOCKS_PER_SEC / 1000) / (double)(1000 * 60 * 60));

                                    if ((i_early_write == 0) && (d_time_elapsed > d_time_stop)) {

                                        i_early_write = 1;

                                        cout << "Early write at " 
                                             << d_time_elapsed << " hours" << endl;

                                        cout << "Last job completed at write: " 
                                             << i_l1 << "," << i_l2 << "," 
                                             << i_l3 << "," << i_l4 << "," 
                                             << i_L << endl;

                                        cout << "Saving kurtosis power spectrum (3,1) to " 
                                             << s_fn_kl31 << endl;
                                        write_vector(vd_kl31, s_fn_kl31);

                                        cout << "Saving ell (L) slices to " << s_fn_ell_out << endl;
                                        write_vector(vd_L, s_fn_ell_out);

                                        int ai_ell_stop[] = {i_l1, i_l2, i_l3, i_l4, i_L};
                                        vector<int> vi_cur_job (ai_ell_stop, 
                                            ai_ell_stop + sizeof(ai_ell_stop) / sizeof(ai_ell_stop[0]) );

                                        cout << "Saving last job to " << s_fn_cur_job << endl;
                                        write_vector(vi_cur_job, s_fn_cur_job);

                                        cout << "Killing slaves..." << endl;

                                        for (int i_rank_out = 1; i_rank_out < i_size; i_rank_out++) {

                                            /* Tell the slave to exit */

                                            MPI_Send(&i_dead, 
                                                     1, 
                                                     MPI_INT, 
                                                     o_status.MPI_SOURCE, 
                                                     DIETAG, 
                                                     MPI_COMM_WORLD);

                                        }

                                        cout << "Exiting..." << endl;

                                        exit(0);

                                    }

                                }

                            }

                        }

                    }

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
            i_l4ind = (int) vd_result[3];
            i_Lind = (int) vd_result[4];

            d_l1 = vd_result[5];
            d_l2 = vd_result[6];
            d_l3 = vd_result[7];
            d_l4 = vd_result[8];
            d_L = vd_result[9];

            d_r_sum_31 = vd_result[10];

/*            cout << "i_l1ind: " << i_l1ind << "i_l2ind: " << i_l2ind
                 << "i_l3ind: " << i_l3ind << "i_l4ind: " << i_l4ind
                 << "i_Lind: " << i_Lind << endl;
*/
            vd_kl31[i_Lind] += d_r_sum_31;

/*            i_all_counter++;

            if (d_r_sum_31 > 0){

                i_sum_counter++;

            }*/

            //cout << "tabulated sums, counters..." << endl;

            /* Tell the slave to exit */

            MPI_Send(&i_dead, 
                     1, 
                     MPI_INT, 
                     o_status.MPI_SOURCE, 
                     DIETAG, 
                     MPI_COMM_WORLD);

            //cout << "finished killing slaves..." << endl;
        }

    } 

    //slave loop

    else {

        vector<int> vi_work(5,0); // il1, il2, il3, il4, iL
        double d_r_sum_31 = 0.0;

        int i_l1ind, i_l2ind, i_l3ind, i_l4ind, i_Lind; 
        double d_l1, d_l2, d_l3, d_l4, d_L;

        MPI_Status o_status;

        vector<double> vd_lnfa = lnfa(10000);

        while (1) {

            /* Receive a message from the master */

            MPI_Recv(&vi_work[0], vi_work.size(), MPI_INT, 0, MPI_ANY_TAG, 
                MPI_COMM_WORLD, &o_status);

            /* Check the tag of the received message */
            if (o_status.MPI_TAG == DIETAG){

                break;

            }

            /* Do the work */

            d_r_sum_31 = 0.0;

            i_l4ind = vi_work[0];
            i_l1ind = vi_work[1];
            i_l2ind = vi_work[2];
            i_l3ind = vi_work[3];
            i_Lind = vi_work[4];

            d_l1 = vd_l[i_l1ind];
            d_l2 = vd_l[i_l2ind];
            d_l3 = vd_l[i_l3ind];
            d_l4 = vd_l[i_l4ind];
            d_L = vd_L[i_Lind];

/*            cout << "job (int): " << i_l1ind << "," << i_l2ind << "," 
                 << i_l3ind << "," << i_l4ind << "," << i_Lind << endl;

            cout << "job (double): " << d_l1 << "," << d_l2 << "," 
                 << d_l3 << "," << d_l4 << "," << d_L << endl;*/

            // kl_31 section

            if ((fabs(d_l1 - d_l2) <= d_l4) && (fabs(d_l1 + d_l2) >= d_l4)){

                if ((fabs(d_l3 - d_L) <= d_l4) && (fabs(d_l3 + d_L) >= d_l4)){

                    d_r_sum_31 += d_tnl * vd_clcurv[i_Lind] 
                                        * (vd_cltt[i_l1ind] + vd_cltt[i_l2ind]) 
                                        * (vd_cltt[i_l3ind] + vd_cltt[i_l4ind]);

                    for (int i_r=0; i_r<i_num_r; i_r++){

                        d_r_sum_31 += (2.0 * d_gnl * vd_dr[i_r] * pow(vd_r[i_r],2) 
                            * vvd_beta[i_l2ind][i_r] * vvd_beta[i_l4ind][i_r] 
                            * (vvd_alpha[i_l1ind][i_r] * vvd_beta[i_l3ind][i_r] + 
                                vvd_alpha[i_l3ind][i_r] * vvd_beta[i_l1ind][i_r]));

                    }

                    d_r_sum_31 *= sqrt((2*d_l1+1)*(2*d_l2+1)*(2*d_l4+1) 
                        / (4 * M_PI))
                        * sqrt((2*d_l3+1)*(2*d_L+1)*(2*d_l4+1) 
                        / (4 * M_PI))
                    * w3j((int) d_l1, (int) d_l2, (int) d_l4, vd_lnfa)
                    * w3j((int) d_l3, (int) d_L, (int) d_l4, vd_lnfa);

                    d_r_sum_31 = pow(d_r_sum_31,2) / vd_cltt[i_l1ind] 
                        / vd_cltt[i_l2ind] / vd_cltt[i_l3ind] / vd_cltt[i_Lind];

/*                    if (d_r_sum_tmp > 1.){

                        cout << "passed triangle inequality? " << d_l1 << "," << d_l2 << "," 
                             << d_L << ": " << (fabs(d_l1 - d_l2) <= d_L) << " && "
                             << (fabs(d_l1 + d_l2) >= d_L) << endl;

                        cout << "w3j for " << d_l1 << "," << d_l2 << "," << d_L << ": " 
                             << w3j((int) d_l1, (int) d_l2, (int) d_L, vd_lnfa) << endl;

                        cout << "passed triangle inequality? " << d_l3 << "," << d_l4 << "," 
                             << d_L << ": " << (fabs(d_l3 - d_l4) <= d_L) << " && "
                             << (fabs(d_l3 + d_l4) >= d_L) << endl;

                        cout << "w3j for " << d_l3 << "," << d_l4 << "," << d_L << ": " 
                             << w3j((int) d_l3, (int) d_l4, (int) d_L, vd_lnfa) << endl;

                        cout << "d_r_sum_tmp for " << d_l1 << "," << d_l2 << "," << d_l3
                             << "," << d_l4 << ","<< d_L << ": " << d_r_sum_tmp << endl;

                    }*/

                    d_r_sum_31 = d_r_sum_31 / (2*d_L + 1) / (2*d_l4 + 1);

                }

            }

            double ad_data[] = {(double)i_l1ind, (double)i_l2ind, 
                        (double)i_l3ind, (double)i_l4ind, (double)i_Lind, 
                        d_l1, d_l2, d_l3, d_l4, d_L, d_r_sum_31};
            vector<double> vd_result (ad_data, 
                    ad_data + sizeof(ad_data) / sizeof(ad_data[0]) );

/*            cout << "Result for " << d_l1 << "," << d_l2 << "," << d_l3 << ","
                 << d_l4 << "," << d_L << ": " << d_r_sum_22 << "," 
                 << d_r_sum_31 << endl;*/

            MPI_Send(&vd_result[0], vd_result.size(), MPI_DOUBLE, 0, 1, 
                MPI_COMM_WORLD);
        }

    }

    MPI_Finalize();

    if (i_rank == 0){

        cout << "calculating time..." << endl;

        cout << endl << "Time for kurtosis power spectra calculation: " 
             << (clock() - o_start) / (double)(CLOCKS_PER_SEC / 1000) 
             << " ms" << endl;

    }

    // Save kurtosis power spectra as numpy array

    if (i_rank == 0){

        cout << "Saving kurtosis power spectrum (3,1) to " 
             << s_fn_kl31 << endl;
        write_vector(vd_kl31, s_fn_kl31);
        cout << "Saving ell (L) slices to " << s_fn_ell_out << endl;
        write_vector(vd_L, s_fn_ell_out);

        if (remove(s_fn_cur_job.c_str()) != 0){

            cout << "Error deleting " << s_fn_cur_job << endl;

        }
        else{

            cout << "Removing " << s_fn_cur_job << " to end calculation..." 
                 << endl;

        }

    }

    if (i_rank == 0){

        cout << "Done!" << endl << endl ;
/*        cout << "i_sum_counter: " << i_sum_counter << endl;
        cout << "i_all_counter: " << i_all_counter << endl;*/

    }

    if (i_rank == 0){

        int i_print_step = i_num_L / 10;

        for (int i=0; i<i_num_L; i+=i_print_step){

                    cout << "ell: " << vd_L[i] << ", kl31_ana: " 
                         << vd_kl31[i] << endl;
        }
    }

    return 0;
}