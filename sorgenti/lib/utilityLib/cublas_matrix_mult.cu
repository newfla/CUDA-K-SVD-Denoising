#include <denoisingLib.h>

using namespace utl;
using namespace svd;
using namespace thrust;

CuBlassMatrixMult::CuBlassMatrixMult(){}

//********************************************************
// Create cuBlas additional data and move data on DEVICE
//*******************************************************
void CuBlassMatrixMult::init(){

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
void CuBlassMatrixMult::finalize(){

    auto start = std::chrono::steady_clock::now();

    c = new Matrix(a->m, b->n, b->n, cVector);
    cublasDestroy(handle);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//*******************
// CuBlas Sgemm call
// output = alfa*op( A )*op( B ) + beta*C,
//******************
utl::Matrix* CuBlassMatrixMult::multiply(){

    init();

    auto start = std::chrono::steady_clock::now();

    float* pointerA = raw_pointer_cast(a->deviceVector->data());
    float* pointerB = raw_pointer_cast(b->deviceVector->data());
    float* pointerC = raw_pointer_cast(cVector->data());

    cublasSgemm(handle, op, op, a->m, b->n, a->n, &alfa, pointerA, a->ld, pointerB, b->ld, &beta, pointerC, b->ld);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();

    return c;
}