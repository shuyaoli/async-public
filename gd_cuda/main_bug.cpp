// g++ -I/usr/local/MATLAB/R2016b/extern/include -L/usr/local/MATLAB/R2016b/bin/glnxa64 main_bug.cpp -o main.o -lmat -lmx

// ./main.o 
// -0.706488 -0.519653 -0.493828 -0.357066
// 0 0 0 0

#include <iostream>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include "mat.h"

using namespace std;

int main()
{
  MATFile *pmat;
  pmat = matOpen("small_dataset_bug.mat", "r");
  // if (pmat == NULL) {
  //   printf("Error reopening file\n");
  //   assert(false);
  // }

  // mxArray* p_alpha, *p_dim, *p_epoch,* p_n,* p_s,* p_x_a,* p_y;

  // auto safe_get_var = [&pmat](mxArray* &p_var, string name_var) {
  //   p_var = matGetVariable(pmat, name_var.c_str());
  //   if (p_var == NULL) {
  //     cout << "Error reading " << name_var << endl;
  //     assert(false);
  //   }
  // };

  // safe_get_var(p_alpha, "alpha");
  // safe_get_var(p_dim, "dim");
  // safe_get_var(p_epoch, "epoch");
  // safe_get_var(p_n, "n");
  // safe_get_var(p_s, "s");
  // // safe_get_var(p_x_a, "x_a");
  // safe_get_var(p_y, "y");

  mxArray* p_x_a = matGetVariable(pmat, "x_a");
  

  // const double alpha = *mxGetPr(p_alpha);
  // const int dim = *mxGetPr(p_dim);
  // const int epoch = *mxGetPr(p_epoch);
  // const int n = *mxGetPr(p_n);
  // const double s = *mxGetPr(p_s);
  const double *x_a = mxGetPr(p_x_a);
  // // By trial and error, it seems this step fails
  // // if size of x_a exceeds 16374 - a weird bug?
  // // It is indeed 16374, not 16384
  
  // double *y = mxGetPr(p_y);

  // mxDestroyArray(p_alpha);
  // mxDestroyArray(p_dim);
  // mxDestroyArray(p_epoch);
  // mxDestroyArray(p_n);
  // mxDestroyArray(p_s);
  mxDestroyArray(p_x_a);
  // mxDestroyArray(p_y);
  
  if (matClose(pmat) != 0) {
    printf("Error closing file");
    return(EXIT_FAILURE);
  }

  // cout << n << ' ' << dim << endl;
  // for (int i = 0; i < dim * n; i++)
  //   cout << x_a[i] << endl;
  cout << x_a[0] <<' '<< x_a[1] <<' ' << x_a[2]<<' ' << x_a[3]<<endl;

  
  double* z_a = NULL;
  z_a = new double[8192]();
  cout << x_a[0] <<' '<< x_a[1] <<' ' << x_a[2]<<' ' << x_a[3]<<endl;
  // double* mean_z = new double [dim] ();
  // cout << x_a[0] <<' '<< x_a[1] <<' ' << x_a[2]<<' ' << x_a[3]<<endl;
  // double* delta_z = new double [dim]();
  // cout << x_a[0] <<' '<< x_a[1] <<' ' << x_a[2]<<' ' << x_a[3]<<endl;

  // double *d_x_a, *d_y, *d_z_a, *d_mean_z;

  // err_chk(cudaMalloc((void**)&d_x_a, sizeof(double) * n * dim));
  // err_chk(cudaMalloc((void**)&d_y, sizeof(double) * n ));
  // err_chk(cudaMalloc((void**)&d_z_a, sizeof(double) * n * dim));
  // err_chk(cudaMalloc((void**)&d_mean_z, sizeof(double) * dim));
  // cout << "CUDA Malloc done\n";
  

  // err_chk(cudaMemcpy(d_x_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  // err_chk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
  // err_chk(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  // err_chk(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
  // cout << "CUDA Memcpy Host to Device Done\n";
  
  // for (int k = 0; k < epoch; k++) {
  //   //calculate gradient
  //   //zi = zmean - alpha * gradient
  //   // update zmean using cuda
  //   for (int ik = 0; ik < n; ik++) {
  //     double dot = 0;
  //     for (int i = 0; i < dim; i++) 
  //       dot += mean_z[i] * x_a[dim * ik + i];
      
  //     for (int c =  0; c < dim; c++) {
  //       z_a[ik*dim+c] = mean_z[c] - alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * mean_z[c]);
  //     }
  //   }

  //   for (int c = 0; c < dim; c++){
  //     double sum = 0;
  //     for (int i = 0; i < n; i++)
  //       sum += z_a[i * dim + c];
  //     mean_z[c] = sum / n ;
  //   }

    
  // }


  
  // for (int i = 0; i < dim; i++)
  //   printf("%.15f\n", mean_z[i]);
  
  return 0;
}
