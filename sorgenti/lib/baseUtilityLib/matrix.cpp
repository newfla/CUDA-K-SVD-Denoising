#include <baseUtilityLib.h>

using namespace baseUtl;
using namespace thrust;

//***********************************************
//  Constructor (old fashion way - data on HOST)
//  input:  + m matrix rows
//          + n matrix columns
//          + ld matrix leading dimension
//          + matrix values (float*) 
//**********************************************
Matrix::Matrix(int m, int n, int ld, float* matrix){

  this->m = m;
  this->n = n;
  this->ld = ld;
  hostVector = new host_vector<float>(matrix, matrix + (m*n));
}

//************************************************
//  Constructor (data on HOST)
//  input:  + m matrix rows
//          + n matrix columns
//          + ld matrix leading dimension
//          + matrix values (host_vector<float*>) 
//***********************************************
Matrix::Matrix(int m, int n, int ld, host_vector<float>* matrix){

  this->m = m;
  this->n = n;
  this->ld = ld;
  hostVector = matrix;
}

//**************************************************
//  Constructor (data on DEVICE)
//  input:  + m matrix rows
//          + n matrix columns
//          + ld matrix leading dimension
//          + matrix values (device_vector<float*>) 
//*************************************************
Matrix::Matrix(int m, int n, int ld, device_vector<float>* matrix){

  this->m = m;
  this->n = n;
  this->ld = ld;
  deviceVector = matrix;
}

//******************************************************
//  Destructor
//  Free data on HOST (data on DEVICE is already freed) 
//*****************************************************
Matrix::~Matrix(){

  if(hostVector != NULL)
    delete hostVector;
}

//***********************************************
//  Create a random values matrix (data on HOST)
//  input:  + m matrix rows
//          + n matrix columns
//          + ld matrix leading dimension
//  output: + matrix (Matrix*) 
//**********************************************
Matrix* Matrix::randomMatrixHost(int m, int n, int ld){

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

//************************************************
//  Clone this matrix HOST to HOST
//  output: + matrix (Matrix*)
//***********************************************
Matrix* Matrix::cloneHost(){

  float* matrix = new float[ld*n]();

  for(int i = 0; i < ld*n; i++)
    matrix[i] = hostVector->data()[i];
  
  return new Matrix(m, n, ld, matrix);
}

//****************************
//  Copy Data HOST --> DEVICE
//***************************
void Matrix::copyOnDevice(){

    if(deviceVector != NULL)
      cudaFree(raw_pointer_cast(deviceVector->data()));

    if(hostVector != NULL)    
      deviceVector = new device_vector<float>(hostVector->begin(), hostVector->end());
}

//****************************
//  Copy Data HOST <-- DEVICE
//***************************
void Matrix::copyOnHost(){

    if(hostVector != NULL)
      delete hostVector;

    if(deviceVector != NULL)
      hostVector = new host_vector<float>(deviceVector->begin(), deviceVector->end());
}
