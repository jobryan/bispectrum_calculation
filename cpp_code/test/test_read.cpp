/*
test_read.cpp

Created on June 2, 2014
Updated on June 3, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Testing code for C++ -- read from file to vector, write from a 
            vector to a file.

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
#include <ctime>
#include <iterator>
#include <algorithm>

using namespace std;

/* Read 1D vectors */

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

    cout << "vd_data size:" << vd_data.size() << endl;

    return vd_data;
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

int main(){

    int i_lmax = 5;
    int i_num_L = 2;

    int ai_data[] = {i_lmax, i_lmax, i_num_L};
    vector<int> vi_dims (ai_data, 
            ai_data + sizeof(ai_data) / sizeof(ai_data[0]) );

    vector<int> vi_tuple;

    for (int i = 0; i<50; i++){

        vi_tuple = cart_index(i, vi_dims);

        cout << "Index: " << i << " Tuple: (" << vi_tuple[0]
             << "," << vi_tuple[1] << "," << vi_tuple[2] << ")" << endl;

    }

/*    vi_tuple = cart_index(15002, vi_dims);

    cout << "Index: " << 15002 << " Tuple: (" << vi_tuple[0]
         << "," << vi_tuple[1] << "," << vi_tuple[2] << ")" << endl;

    vi_tuple = cart_index(15002, vi_dims);

    cout << "Index: " << 15002 << " Tuple: (" << vi_tuple[0]
         << "," << vi_tuple[1] << "," << vi_tuple[2] << ")" << endl;*/

/*
    vector<double> vd_lnfa = lnfa(10000);

    int ai_data[] = {1300,1350,1400,1450};
    vector<int> vi_data (ai_data, 
            ai_data + sizeof(ai_data) / sizeof(ai_data[0]) );

    for (int i=0; i<4; i++){

        for (int j=0; j<4; j++){

            for (int k=0; k<4; k++){

                cout << "w3j(" << vi_data[i] << "," << vi_data[j] << "," 
                     << vi_data[k] << ") = "
                     << w3j(vi_data[i], vi_data[j], vi_data[k], vd_lnfa) 
                     << endl;

            }

        }

    }*/

/*    clock_t o_start;
    o_start = clock();

    string s_fn_cltt = "../../output/na_cltt_ell_1499.txt";
    vector<double> vd_cltt = read_vector(s_fn_cltt);

    cout << "Time to load vd_cltt: " 
         << (clock() - o_start) / (double)(CLOCKS_PER_SEC / 1000) 
         << " ms" << endl;

    o_start = clock();

    string s_fn_alpha = "../../data/alpha_ell_1499_r_437.txt";
    string s_fn_beta = "../../data/beta_ell_1499_r_437.txt";

    vector< vector<double> > vvd_alpha = read_matrix(s_fn_alpha);
    vector< vector<double> > vvd_beta = read_matrix(s_fn_beta);

    for (int i=0; i<3; i++){

        for (int j=0; j<3; j++){

            cout << "i,j:" << i << "," << j << " alpha:" << vvd_alpha[i][j]
                 << " beta:" << vvd_beta[i][j] << endl;;

        }

    }*/


/*    string s_fn_mll = "../../output/na_mll_ell_2000.txt";
    vector< vector<double> > vvd_mll = read_matrix(s_fn_mll);

    cout << "Time to load vvd_mll: " 
         << (clock() - o_start) / (double)(CLOCKS_PER_SEC / 1000) 
         << " ms" << endl;

    vector<vector<vector<double> > > vvvd_example = test_cube(3);
    vvvd_example[0][0][0] = 1;
    vvvd_example[1][1][1] = 1;
    vvvd_example[2][2][2] = 1;
    string s_fn = "test_write.txt";

    write_3dvector(vvvd_example, s_fn);

    int i_size = 3;

    vector<vector<vector<double> > > vvvd_read = read_3dvector(s_fn, i_size);

    for (int i=0; i<i_size; i++){

        for (int j=0; j<i_size; j++){

            for (int k=0; k<i_size; k++){

                cout << "i,j,k: " << i << "," << j << "," << k << " value:" 
                     << vvvd_read[i][j][k] << endl;

             }

        }

    }
*/

/*    int i_size2 = 40;
    string s_fn_bi_ana = "../output/vvvd_bi_ana_40_rsteps_40_ellsteps.dat";

    vector<vector<vector<double> > > vvvd_read2 = read_3dvector(s_fn_bi_ana, 
                                                                i_size2);

    cout << "Loaded bi_ana with shape (" << vvvd_read2.size() << "," 
        << vvvd_read2[0].size() << "," << vvvd_read2[0][0].size() << ")" 
        << endl;

    for (int i=0; i<3; i++){

        for (int j=0; j<3; j++){

            for (int k=0; k<3; k++){

                cout << "i,j,k: " << i << "," << j << "," << k << " value:" 
                     << vvvd_read2[i][j][k] << endl;

             }

        }

    }
*/

    return 0;
}