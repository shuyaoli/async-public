#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <mex.h>

using namespace std;

double sig (double x) {
  return 1/(1 + exp(-x));
}

double dot (double* x, double* y, int dim) {
  double result = 0;
  for (int i = 0; i < dim; i++) 
    result += x[i] * y[i];
  return result;
}

double* vector_sum (double* x, double *y, int dim) {
    
  double* result = new double[dim];
  for (int i = 0; i < dim; i++)
    result[i] = x[i] + y[i];
  return result;
}

double* vector_difference(double* x, double *y, int dim) {
    
  double* result = new double[dim];
  for (int i = 0; i < dim; i++)
    result[i] = x[i] - y[i];
  return result;
}

void* vector_increment(double* x, double* incr, int dim){
  for (int i = 0; i < dim; i++)
    x[i] += incr[i];
}

// double fi(double w[], double xi[], double yi, double s, int dim) {
// Note that type need to be changed when uncommenting this function
//     return -log(sig(-yi * dot(w, xi))) + s/2 * dot (w, w); 
// }

double* grad_fi(double* w, double* xi, double yi, double s, int dim) {

  double* result = new double[dim];
    
  for (int j = 0; j < dim; j++) {
    result[j] = -sig( -yi * dot(w, xi, dim)) * yi * xi[j] + s * w[j];
  }
    
  return result;
}

double** array2rowvectors (const double *array, int num_row, int num_col) {
  // return a vector of row vectors
  // Test: 
  //     input {1,2,3,4,5,6}, 
  //     output 
  //            1 3 5
  //            2 4 6
  
  double** matrix = new double* [num_row];
  for (int r = 0; r < num_row; r++) {
    matrix[r] = new double[num_col];
    for (int c = 0; c < num_col; c++) {
      matrix[r][c] = array[c * num_row + r];
    }
  }

  return matrix;
}

double* mean_rowvectors(double** x, int num_row, int num_col) {

  double* result = new double[num_col];
  
  for (int c = 0; c < num_col; c++) {
    double s = 0;
    for (int r = 0; r < num_row; r++)
      s += x[r][c];
    result[c] = s / num_row;
  }
  return result;
}

double* scalar_vector_product (double c, double* v, int dim) {
  double* result = new double[dim];
  for (int i = 0; i < dim; i++)
    result[i] = c * v[i];
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

    
  double** x_v = array2rowvectors (x, n, dim);
  double** phi_v = array2rowvectors (phi, n, dim);
    
  double** grad_v = new double* [n];
  for (int r = 0; r < n; r++) 
    grad_v[r] = grad_fi(phi_v[r], x_v[r], y[r], s, dim);
  
  double* mean_phi = mean_rowvectors(phi_v, n, dim);
    
  srand(1);
  for (int k = 0; k < n * epoch; k++) {
    // Pick j
    int j = rand() % n;

    //
    double* mean_grad = mean_rowvectors(grad_v, n, dim);
    double* term2 = scalar_vector_product(-1.0/alpha/s, mean_grad, dim);
    double* w = vector_sum(mean_phi, term2, dim);
    delete[] term2;
    
    
    double* dif = vector_difference(w, phi_v[j], dim);
    double* incr = scalar_vector_product(1.0/n, dif, dim);
    delete[] dif;
    
    vector_increment(mean_phi, incr, dim);
    delete[] incr;
    
    phi_v[j] = w;
    grad_v[j] = grad_fi(phi_v[j], x_v[j], y[j], s, dim);
    
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
