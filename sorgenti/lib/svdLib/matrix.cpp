#include <svdLib.h>

using namespace svd;
using namespace thrust;

Matrix::Matrix(int m, int n, int ld, float* matrix){
  this->m = m;
  this->n = n;
  this->ld = ld;
  hostVector = new host_vector<float>(matrix, matrix + (m*n));
}

Matrix::~Matrix(){
  if(hostVector != NULL)
    delete hostVector;
  
}

Matrix* Matrix::randomMatrix(int m, int n, int ld){
  float* matrix = new float[ld*n]();
  Matrix* out;
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<float> dist(0.,1.);

  for(int i = 0; i < ld*n; i++)
    matrix[i] = dist(engine);

  out = new Matrix(m, n, ld, matrix);

  return out;
}

Matrix* Matrix:: clone(){
  float* matrix = new float[ld*n]();

  for(int i = 0; i < ld*n; i++)
    matrix[i] = hostVector->data()[i];
  
  return new Matrix(m, n, ld, matrix);
}
