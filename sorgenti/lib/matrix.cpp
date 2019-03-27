#include <svdLib.h>

using namespace svd;

Matrix::Matrix(int m, int n, int ld, double* matrix){
  this->m = m;
  this->n = n;
  this->ld = ld;
  this->matrix = matrix;
}

Matrix::~Matrix(){
  //std::cout<<"\nMatrix\n";
  delete[] matrix;
}

Matrix* Matrix::randomMatrix(int m, int n, int ld){
  double* matrix = new double[ld*n]();
  Matrix* out = new Matrix(m, n, ld, matrix);
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<float> dist(0.,1.);

  for(int i = 0; i < n*m; i++)
    out->matrix[i] = dist(engine);

  return out;
}
