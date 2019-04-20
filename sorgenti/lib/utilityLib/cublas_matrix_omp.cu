#include <utilityLib.h>

using namespace utl;
using namespace thrust;

CuBlasMatrixOmp::CuBlasMatrixOmp(){}

//**************************************
//  Set eps_float maxIter
//  input: epsilon maxIter (float, int as float)
//*************************************
void CuBlasMatrixOmp::setLimits(float epsilon, int maxIters){
    
    this->epsilon = epsilon;
    this->maxIters = (int) maxIters;
}

//********************************************************
// Create cuBlas additional data and move data on DEVICE
//*******************************************************
void CuBlasMatrixOmp::init(){

    auto start = std::chrono::steady_clock::now();

    if(a->deviceVector == NULL)
        a->copyOnDevice();

    if(b->deviceVector == NULL)
        b->copyOnDevice();

    sparseCode = new device_vector<float>(a->n * b->n);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//******************************
// Clear cuBlas additional data
//*****************************
void CuBlasMatrixOmp::finalize(){

    auto start = std::chrono::steady_clock::now();

    c = new Matrix(b->n, a->n, b->n, sparseCode);
    cublasDestroy(handle);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}


// **********************************************************************
// MATCHING PURSUIT ALGORITM (OMP) FOR SPARSE CODING 
// Input:  + patchesMatrix [feature per patch * number of patches] 
//         + dictionary   [feature * atoms]         
// Output: + sparse coding of input vector [ atoms * number of patches]
//**********************************************************************
utl::Matrix* CuBlasMatrixOmp::work(Matrix* patchesMatrix, Matrix* dictionaryMatrix){

    this->a = a;
    this->b = b;

    init();

    auto start = std::chrono::steady_clock::now();

    float* noisePatches = raw_pointer_cast(a->deviceVector->data());
    float* dictionary = raw_pointer_cast(b->deviceVector->data());
    float* sparseCodeV = raw_pointer_cast(sparseCode->data());
    float normi, normf, iterEpsilon, coff;
    int q;

    for(int i = 0; i < patchesMatrix->n; i++){ //n == #columns == # patches   

        cublasSgemv(handle,
                    CUBLAS_OP_T,
                    dictionaryMatrix->m,
                    dictionaryMatrix->n,
                    &alfa,
                    dictionary,
                    dictionaryMatrix->ld,
                    noisePatches + i * patchesMatrix->m,
                    1,
                    &beta,
                    sparseCodeV + i * dictionaryMatrix->n,
                    1);
    }

    for(int i = 0; i < patchesMatrix->n; i++){ //n == #columns == # patches

        cublasSnrm2(handle,
                    dictionaryMatrix->n,
                    sparseCodeV + i * dictionaryMatrix->n,
                    1,
                    &normi);

        int iter = 0;

        iterEpsilon = sqrtf( epsilon * normi);
        normf = normi;

        while(normf > iterEpsilon && iter < maxIters){
            
          cublasSgemv(handle,
                    CUBLAS_OP_N,
                    dictionaryMatrix->m,
                    dictionaryMatrix->n,
                    &alfa,
                    dictionary,
                    dictionaryMatrix->ld,
                    sparseCodeV + i * dictionaryMatrix->n,
                    1,
                    &beta,
                    noisePatches + i * patchesMatrix->m,
                    1);

            cublasIsamax(handle,
                        dictionaryMatrix->m,
                        noisePatches + i * patchesMatrix->m,
                        1,
                        &q);

            q-=1;
            
            cublasGetVector(1, 
                            sizeof(coff),
                            noisePatches + q + i * patchesMatrix->m,
                            1,
                            &coff,
                            1);
            
            coff = - coff;

            cublasSaxpy(handle,
                        dictionaryMatrix->n,
                        &coff,
                        dictionary + q,
                        dictionaryMatrix->m,
                        sparseCodeV + i * dictionaryMatrix->n,
                        1);

            cublasSnrm2(handle,
                        dictionaryMatrix->n,
                        sparseCodeV + i * dictionaryMatrix->n,
                        1,
                        &normi);
            
            iter++;

        }
        
    }
    

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();

    return c;
}