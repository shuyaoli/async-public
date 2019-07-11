#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#define n 4096
#define dim 32
#define s 1
#define epoch 30
#define alpha 0.5
#define SIZE "SMALL"

using namespace std;

// void err_chk(cudaError err) {
//   if (err != cudaSuccess) {
//     cout << cudaGetErrorString(err) << endl;
//     assert(false);
//   }
// }

void read_var(double* var, string var_name, int len)
{
  string filename = string(SIZE) + string("/") + var_name + string(".txt");
  ifstream var_file(filename);
  string line;
  if (!var_file.is_open()) {
    cout << "Failed to open " << var_name << endl;
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < len; i++) {
    if (getline(var_file, line)) 
      var[i] = stod(line);
    else {
      cout << "Error loading " << var_name << endl;
      exit(EXIT_FAILURE);
    }
  }
  var_file.close();
}


  
int main()
{
  double *x_a = new double [n * dim];
  double *y = new double [dim];

  read_var(x_a, "x_a", n * dim);
  read_var(y, "y", dim);

  
  double* z_a =  new double[n * dim]();

  double* mean_z = new double [dim] ();

  double* delta_z = new double [dim]();


  // // double *d_x_a, *d_y, *d_z_a, *d_mean_z;

  // // err_chk(cudaMalloc((void**)&d_x_a, sizeof(double) * n * dim));
  // // err_chk(cudaMalloc((void**)&d_y, sizeof(double) * n ));
  // // err_chk(cudaMalloc((void**)&d_z_a, sizeof(double) * n * dim));
  // // err_chk(cudaMalloc((void**)&d_mean_z, sizeof(double) * dim));
  // // cout << "CUDA Malloc done\n";
  

  // // err_chk(cudaMemcpy(d_x_a, x_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  // // err_chk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
  // // err_chk(cudaMemcpy(d_z_a, z_a, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
  // // err_chk(cudaMemcpy(d_mean_z, mean_z, sizeof(double) * dim, cudaMemcpyHostToDevice));
  // // cout << "CUDA Memcpy Host to Device Done\n";

  
  for (int k = 0; k < epoch; k++) {
    //calculate gradient
    //zi = zmean - alpha * gradient
    // update zmean using cuda
    for (int ik = 0; ik < n; ik++) {
      double dot = 0;
      for (int i = 0; i < dim; i++) 
        dot += mean_z[i] * x_a[dim * ik + i];
      
      for (int c =  0; c < dim; c++) {
        z_a[ik*dim+c] = mean_z[c] - alpha * (-1.0 / (1+exp(y[ik] * dot)) * y[ik] * x_a[dim * ik + c] + s * mean_z[c]);
      }
    }

    for (int c = 0; c < dim; c++){
      double sum = 0;
      for (int i = 0; i < n; i++)
        sum += z_a[i * dim + c];
      mean_z[c] = sum / n ;
    }

    cout << mean_z[0];
  }


  
  for (int i = 0; i < dim; i++)
    printf("%.15f\n", mean_z[i]);
  
  return 0;
}
