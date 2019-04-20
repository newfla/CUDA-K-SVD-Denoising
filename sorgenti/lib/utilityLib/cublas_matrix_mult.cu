#include <utilityLib.h>

using namespace utl;
using namespace thrust;

CuBlasMatrixMult::CuBlasMatrixMult(){}

//********************************************************
// Create cuBlas additional data and move data on DEVICE
//*******************************************************
void CuBlasMatrixMult::init(){

    auto start = std::chrono::steady_clock::now();

    if(a->deviceVector == NULL)
        a->copyOnDevice();

    if(b->deviceVector == NULL)
        b->copyOnDevice();

    cVector = new device_vector<float>(a->m * b->n);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//******************************
// Clear cuBlas additional data
//*****************************
void CuBlasMatrixMult::finalize(){

    auto start = std::chrono::steady_clock::now();

    c = new Matrix(a->m, b->n, b->n, cVector);
    cublasDestroy(handle);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}


//************************************************
// Set cuBlas ops
// input = 2x operation type  (cublasOperation_t) 
//***********************************************
void CuBlasMatrixMult::setOps(cublasOperation_t op1, cublasOperation_t op2){

    this->op1 = op1;
    this->op2 = op2;
}

//*******************
// CuBlas Sgemm call
// output = alfa*op( A )*op( B ) + beta*C,
//******************
utl::Matrix* CuBlasMatrixMult::work(Matrix* a, Matrix* b){

    this->a = a;
    this->b = b;

    init();

    auto start = std::chrono::steady_clock::now();

    float* pointerA = raw_pointer_cast(a->deviceVector->data());
    float* pointerB = raw_pointer_cast(b->deviceVector->data());
    float* pointerC = raw_pointer_cast(cVector->data());

    cublasSgemm(handle,
                op1,
                op2,
                a->m,
                b->n,
                a->n,
                &alfa,
                pointerA,
                a->ld,
                pointerB,
                b->ld,
                &beta,
                pointerC,
                b->ld);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();

    return c;
}