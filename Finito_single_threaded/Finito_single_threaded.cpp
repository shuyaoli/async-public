#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <mex.h>

using namespace std;

double sig (double x) {
    return 1/(1 + exp(-x));
}

double dot (const vector <double> &x, const vector <double> &y) {
    double result = 0;
    int dim = x.size();
    for (int i = 0; i < dim; i++) 
        result += x[i] * y[i];
    return result;
}

vector <double> vector_sum (const vector <double> &x, 
        const vector <double> &y) {
    
  vector <double> result;
  for (int i = 0; i < x.size(); i++)
    result.push_back(x[i] + y[i]);
  return result;
}

// double fi(double w[], double xi[], double yi, double s, int dim) {
// Note that type need to be changed when uncommenting this function
//     return -log(sig(-yi * dot(w, xi))) + s/2 * dot (w, w); 
// }

vector <double> grad_fi(const vector<double> & w, const vector <double> & xi, double yi, double s) {
    int dim = xi.size();
    vector <double> result;
    
    for (int j = 0; j < dim; j++) {
        double each = -sig( -yi * dot(w, xi)) * yi * xi[j] + s * w[j];
        result.push_back(each);
    }
    
    return result;
}

vector < vector <double> > array2rowvectors (const double *array, int num_row, int num_col) {
  // return a vector of row vectors
  // Test: 
  //     input {1,2,3,4,5,6}, 
  //     output 
  //            1 3 5
  //            2 4 6
    
  vector < vector <double > > result;
  for (int r = 0; r < num_row; r++) {
    vector <double> row;
    for (int c = 0; c < num_col; c++) 
      row.push_back(array[c * num_row + r]);
    result.push_back(row);
  }
  return result;
  
}

vector <double> sum_rowvectors( const vector < vector <double> > & x) {
  int num_row = x.size();
  int num_col = x[0].size();
  vector <double> result;
  
  for (int c = 0; c < num_col; c++) {
    double s = 0;
    for (int r = 0; r < num_row; r++)
      s += x[r][c];
    result.push_back(s);
  }
  return result;
}

vector <double> scalar_vector_product (double c, const vector <double> & v) {
  vector <double> result;
  for (int i = 0; i < v.size(); i++)
    result.push_back(c * v[i]);
  return result;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
//     Input: x, y, initial random weight phi, alpha, s, epoch
//     Output: trained decision boundary
    const int n = mxGetDimensions(prhs[0])[0];
    const int dim = mxGetDimensions(prhs[0])[1];
    
    const double* x = mxGetPr(prhs[0]);
    const double* y = mxGetPr(prhs[1]);
    const double* phi = mxGetPr(prhs[2]);
    const double alpha = *mxGetPr(prhs[3]);
    const double s = *mxGetPr(prhs[4]);
    const int epoch = *mxGetPr(prhs[5]);
    
    
    vector < vector <double> > x_v = array2rowvectors (x, n, dim);
    vector < vector <double> > phi_v = array2rowvectors (phi, n, dim);
    
    
    vector < vector <double> > p_v;
    for (int i = 0; i < n; i++) {
        vector <double> term1 = grad_fi(phi_v[i], x_v[i], y[i], s) ;
        vector <double> term2 = scalar_vector_product(-alpha * s, phi_v[i]);
        vector <double> row = vector_sum(term1, term2) ;
        p_v.push_back(row);
    }
    
    srand(1);
    for (int k = 0; k < n * epoch; k++) {
        vector <double> sum_p = sum_rowvectors(p_v);
        vector <double> w = scalar_vector_product(-1/(alpha * s * n), sum_p);
        int j = rand() % n;
        phi_v[j] = w;
        vector <double> term1 = grad_fi(phi_v[j], x_v[j], y[j], s) ;
        vector <double> term2 = scalar_vector_product(-alpha * s, phi_v[j]);
        p_v[j] = vector_sum(term1, term2);
    }
    
    // Output 
    double * array = new double [n * dim];
    
    plhs[0] = mxCreateDoubleMatrix(n, dim, mxREAL);
    double * ptr = mxGetPr(plhs[0]);

    for (int r = 0; r < n; r++) {
      for (int c = 0; c < dim; c++)
        ptr[c * n + r] = /**/ phi_v /**/ [r][c]; //It suffices to change this line to change output
    }
    
    return;
   
}