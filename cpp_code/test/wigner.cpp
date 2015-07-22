/*
wigner.cpp

Created on June 5, 2014
Updated on June 5, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Wigner-3j symbol calculations (using Stirling's approximation for
            factorial).

@command:   Needs to be made by:

            make wigner.cpp

            then run

            ./wigner

*/

// C++ imports
#include <iostream> // cout
#include <vector>
#include <math.h> // fabs, exp, log, sqrt, pow

using namespace std;

vector<double> lnfa(int i_max){

    vector<double> vd_lnfa; //natural log of factorial
    vd_lnfa.push_back(0.0);
    vd_lnfa.push_back(0.0);

    for (int i=2; i<10000; i++){

        vd_lnfa.push_back(log(i) + vd_lnfa[i-1]);

    }

        return vd_lnfa;

}


double w3j(int i_l1, int i_l2, int i_l3){
    /* 
        Calculates the wigner-3j symbol. Please see eq. 15 of 
        http://mathworld.wolfram.com/Wigner3j-Symbol.html for more
        details. 
    */
    
    vector<double> vd_lnfa = lnfa(10000);

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

int main(){

    cout.precision(16);
    cout << "w3j(1,2,3) = " << w3j(1,2,3) << endl;
    cout << "w3j(2,3,5) = " << w3j(2,3,5) << endl;
    cout << "w3j(1,3,4) = " << w3j(1,3,4) << endl;
    cout << "w3j(2,2,4) = " << w3j(2,2,4) << endl;
    cout << "w3j(3,3,6) = " << w3j(3,3,6) << endl;
    cout << "w3j(1000,1200,1300) = " << w3j(1000,1200,1300) << endl;

}
