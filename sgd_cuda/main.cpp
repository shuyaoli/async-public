#include <iostream>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include <string>
#include "mat.h"

using namespace std;

void err_chk(cudaError err) {
  if (err != cudaSuccess) {
    cout << cudaGetErrorString(err) << endl;
    assert(false);
  }
}

int main()
{
  MATFile *pmat;
  pmat = matOpen("small_dataset.mat", "r");
  if (pmat == NULL) {
    printf("Error reopening file\n");
    assert(false);
  }

  mxArray* p_alpha, *p_dim, *p_epoch,* p_n,* p_s,* p_x,* p_y;

  auto safe_get_var = [&pmat](mxArray* &p_var, string name_var) {
    p_var = matGetVariable(pmat, name_var.c_str());
    if (p_var == NULL) {
      cout << "Error reading " << name_var << endl;
      assert(false);
    }
  };

  safe_get_var(p_alpha, "alpha");
  safe_get_var(p_dim, "dim");
  safe_get_var(p_epoch, "epoch");
  safe_get_var(p_n, "n");
  safe_get_var(p_s, "s");
  safe_get_var(p_x, "x");
  safe_get_var(p_y, "y");


  // p_y = matGetVariable(pmat, "y");
  // if (p_y == NULL) {
  //   printf("Error reading y\n");
  //   assert(false);
  // }
  const double alpha = *mxGetPr(p_alpha);
  const int dim = *mxGetPr(p_dim);
  const int epoch = *mxGetPr(p_epoch);
  const int n = *mxGetPr(p_n);
  const double s = *mxGetPr(p_s);
  double *x = mxGetPr(p_x);
  double *y = mxGetPr(p_y);

  mxDestroyArray(p_alpha);
  mxDestroyArray(p_dim);
  mxDestroyArray(p_epoch);
  mxDestroyArray(p_n);
  mxDestroyArray(p_s);
  mxDestroyArray(p_x);
  mxDestroyArray(p_y);
  
  if (matClose(pmat) != 0) {
    printf("Error closing file");
    return(EXIT_FAILURE);
  }
  
  return 0;
}
