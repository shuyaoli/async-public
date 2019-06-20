#include <iostream>
#include <cmath>
#include <cstdlib>

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

double** array2rowvectors (double *array, int num_row, int num_col) {
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


int main()
{
  double array[] = {1,2,3,4,5,6};
  int num_col = 3;
  int num_row = 2;

  double **matrix = array2rowvectors(array, num_row, num_col);
  
  for (int i = 0; i < num_row; i++) {
    for (int c = 0; c < num_col; c++)
      cout << matrix[i][c] << ',';
    cout << '\n';
  }

 
  return 0;
}
