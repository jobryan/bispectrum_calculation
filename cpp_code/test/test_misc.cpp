/*
test_misc.cpp

Created on June 3, 2014
Updated on June 3, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Testing code for C++ -- read from file to vector

@command:   Needs to be made by:

            make test_read

            then run

            ./test_read

*/

// C++ imports
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h> // fabs
#include <tuple>
#include <algorithm>
#include <regex>

using namespace std;

/* Load 1D vectors */

vector<double> read_vector(string s_fn){

    fstream o_ifs(s_fn.c_str());

    string s_line;

    vector<double> vd_data;

    while(getline(o_ifs, s_line)){

        vd_data.push_back(atof(s_line.c_str()));
    }

    cout << "vd_data size:" << vd_data.size() << endl;

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


/* Load 2D vectors */

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

    cout << "vd_data size:" << vd_data.size() << endl;

    return vd_data;
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
Select subarrays
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

            cout << "selecting row " << i_ind1 << " and column " << i_ind2 << endl;
            cout << "value: " << vt_tmp[i_ind2] << endl;

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
Equivalent to Python's xrange(first, last, inc)
*/

vector<int> xrange(int i_first, int i_last, int i_incrememt=1){

    vector<int> vi_return;

    for (int i=i_first; i<i_last+1; i+=i_incrememt){

        vi_return.push_back(i);

    }

    return vi_return;

}

int main(){

vector<double> vd_L;
vector<int> vi_L_ind;

vd_L = spaced_vector(1,100,100,"lin");

for (int i=0; i<vd_L.size(); i++){

    cout << "vd_L (original): " << vd_L[i] << endl;

}

cout << "-------------------------------" << endl;

tie(vd_L, vi_L_ind) = sub_arr(vd_L, 10, (double)100, "log");

for (int i=0; i<vd_L.size(); i++){

    cout << "vd_L (log): " << vd_L[i] << endl;

}


/*    vector<double> vd_source = spaced_vector(2, 10, 5, "log");

    for (int i=0; i<vd_source.size(); i++){

        cout << "vd_source (log): " << vd_source[i] << endl;

    }

    cout << "-------------------------------" << endl;

    vd_source = spaced_vector(2, 100, 10, "log");

    for (int i=0; i<vd_source.size(); i++){

        cout << "vd_source (log): " << vd_source[i] << endl;

    }*/

/*    vector<double> vd_return;
    vector<int> vi_ind;

    static const double ad_target[] = {16,2,77,29,42,64};
    vector<double> vd_target (ad_target, 
                        ad_target + sizeof(ad_target) / sizeof(ad_target[0]) );

    tie(vd_return, vi_ind) = closest_points(vd_source, vd_target);

    for (int i=0; i<vd_return.size(); i++){

        cout << "vd_return: " << vd_return[i] << endl;
        cout << "vi_ind: " << vi_ind[i] << endl;

    }

    vector<double> vd_sub;
    vector<int> vi_ind2;

    tie(vd_sub, vi_ind2) = sub_arr(vd_source, 3, 50);

    for (int i=0; i<vd_sub.size(); i++){

        cout << "vd_sub: " << vd_sub[i] << endl;
        cout << "vi_ind2: " << vi_ind2[i] << endl;

    }*/

/*    string s_fn_mll = "../output/na_mll_ell_2000.txt";
    vector< vector<double> > vvd_mll = read_matrix(s_fn_mll);

    for (int i=0; i<5; i++){

        for (int j=0; j<5; j++){

            cout << "matrix element: " << i << "," << j << " " 
                 << vvd_mll[i][j] << endl;

        }

    }


    vector<double> vd_ind1 = spaced_vector(0, 100, 10, "lin");
    vector<int> vi_ind1(vd_ind1.begin(), vd_ind1.end());
*/
/*    for (int i=0; i<vi_ind1.size(); i++){
        cout << vi_ind1[i] << endl;
    }*/

/*    vector<double> vd_ind2 = spaced_vector(0, 100, 10, "lin");
    vector<int> vi_ind2(vd_ind2.begin(), vd_ind2.end());

    vector< vector<double> > vvd_mll_slice = slice_arr(vvd_mll, vi_ind1, 
                                                       vi_ind2);

    for (int i=0; i<vvd_mll_slice.size(); i++){

        for (int j=0; j<vvd_mll_slice[0].size(); j++){

            cout << "matrix element: " << vi_ind1[i] << "," << vi_ind2[j] << " " 
                 << vvd_mll_slice[i][j] << endl;

        }

    }*/

/*    int i_num_cores = 2;

    vector<int> vi_1 = xrange(1, 2, i_num_cores);
    vector<int> vi_2 = xrange(0, 2, 1);
    vector<int> vi_3 = xrange(0, 2, 1);

    vector<vector<int> > vvd_tuples = cart_triple_prod(vi_1, vi_2, vi_3);

    for (int i=0; i<vvd_tuples.size(); i++){

        cout << "tuple: " << i << ", (" << vvd_tuples[i][0]  << "," 
             << vvd_tuples[i][1] << "," << vvd_tuples[i][2] << ")" << endl;

    }
*/
    return 0;

}