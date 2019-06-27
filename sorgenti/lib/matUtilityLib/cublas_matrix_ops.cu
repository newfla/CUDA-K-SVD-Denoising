#include <matUtilityLib.h>

using namespace matUtl;
using namespace baseUtl;

cublasHandle_t* CuBlasMatrixOps::handle = NULL;
CuBlasMatrixOps::CuBlasMatrixOps(){}

//************************************************
// Set cuBlas ops
// input = 2x operation type  (cublasOperation_t) 
//***********************************************
void CuBlasMatrixOps::setOps(cublasOperation_t op1, cublasOperation_t op2){

    this->op1 = op1;
    this->op2 = op2;
}

//********************************************************
// Create cuBlas additional data and move data on DEVICE
//*******************************************************
void CuBlasMatrixOps::init(){
    if(handle == NULL){   
        handle = new cublasHandle_t();
        cublasCreate(handle);
    }

    if(a->deviceVector == NULL)
        a->copyOnDevice();

    if(b->deviceVector == NULL)
        b->copyOnDevice();

} 

//******************************
// Clear cuBlas additional data
//*****************************
void CuBlasMatrixOps::finalize(){
    if(handle != NULL){
        cublasDestroy(*handle);
        handle = NULL;
    }
}