#include <matUtilityLib.h>

using namespace baseUtl;
using namespace svd;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;

CuBlasMatrixOmp::CuBlasMatrixOmp(){}

//********************************************************
// Create cuBlas additional data and move data on DEVICE
//*******************************************************
void CuBlasMatrixOmp::init(){

    auto start = std::chrono::steady_clock::now();

    cublasCreate(&handle);

    if(a->deviceVector == NULL)
        a->copyOnDevice();

    if(b->deviceVector == NULL)
        b->copyOnDevice();

    sparseCode = new device_vector<float>(b->n * a->n,0);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//******************************
// Clear cuBlas additional data
//*****************************
void CuBlasMatrixOmp::finalize(){

    auto start = std::chrono::steady_clock::now();

    c = new Matrix(a->n, b->n, a->n, sparseCode);
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
baseUtl::Matrix* CuBlasMatrixOmp::work(Matrix* patchesMatrix, Matrix* dictionaryMatrix){

    this->b = patchesMatrix;
    this->a = dictionaryMatrix;
    init();
    

    auto start = std::chrono::steady_clock::now();
    
    float norm = 0;
    int max = 0, min = 0, chosenAtomIdx = 0;

    for(int inputIdx = 0; inputIdx < patchesMatrix->n; inputIdx++){ //n == #columns == # patches

        device_vector<float> residualVec(patchesMatrix->deviceVector->begin() + (inputIdx * patchesMatrix->m),
                                     patchesMatrix->deviceVector->begin() + ((inputIdx+1) * patchesMatrix->m));
        
        device_vector<float> tempVec(dictionaryMatrix->m);
        device_vector<float> thisSparseCode(dictionaryMatrix->n);
        device_vector<int> chosenAtomIdxList;
		device_vector<float> chosenAtomList; 
        int iter = 0;
        
        while(iter < maxIters){
        
            device_vector<float> proj(dictionaryMatrix->n);

            cublasSgemv(handle,
                        CUBLAS_OP_T,
                        dictionaryMatrix->m,
                        dictionaryMatrix->n,
                        &alfa,
                        raw_pointer_cast(dictionaryMatrix->deviceVector->data()),
                        dictionaryMatrix->ld,
                        raw_pointer_cast(residualVec.data()),
                        1,
                        &beta,
                        raw_pointer_cast(proj.data()),
                        1);

            cublasIsamax(handle,
                         dictionaryMatrix->n,
                         raw_pointer_cast(proj.data()),
                         1,
                         &max);
            max--;

            cublasIsamin(handle,
                         dictionaryMatrix->n,
                         raw_pointer_cast(proj.data()),
                         1,
                         &min);
            min--;
            
            chosenAtomIdx = (abs(proj[max]) > abs(proj[min])) ? max : min;
          
            chosenAtomIdxList.push_back(chosenAtomIdx);

            chosenAtomList.insert(chosenAtomList.end(),
                                  dictionaryMatrix->deviceVector->begin() + (chosenAtomIdx * dictionaryMatrix->m),
                                  dictionaryMatrix->deviceVector->begin() + ((chosenAtomIdx + 1) * dictionaryMatrix->m));
            
            //ChosenAtomList Pseudo-Inverse
            device_vector<float>* copiedList = new device_vector<float> (chosenAtomList.begin(), chosenAtomList.end());
            Matrix* toPinvert = new Matrix(dictionaryMatrix->m, chosenAtomIdxList.size(), dictionaryMatrix->m, copiedList);

            SvdContainer* container = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
            container->setMatrix(toPinvert);
            host_vector<Matrix*> usv = container->getDeviceOutputMatrices();

            device_vector<float> sVector(dictionaryMatrix->m * chosenAtomIdxList.size(),0);            
            device_vector<float> tempMatMult(chosenAtomIdxList.size() * dictionaryMatrix->m);
            device_vector<float> pseudoInverse(chosenAtomIdxList.size() * dictionaryMatrix->m);

            host_vector<int> indicesHost(usv[1]->n);

            for (int i = 0; i < usv[1]->n; i++)
                indicesHost[i] = (i * chosenAtomIdxList.size()) + i;
            
            device_vector<int> indicesDevice = indicesHost;
            
            transform_if(usv[1]->deviceVector->begin(),
                         usv[1]->deviceVector->end(),
                         make_permutation_iterator(sVector.begin(),indicesDevice.begin()),
                         1./_1,
                         not_zero());

            cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N, //sVector è già trasposto a mano in realtà
                usv[2]->m,
                usv[0]->m,
                usv[2]->n,
                &alfa,
                raw_pointer_cast(usv[2]->deviceVector->data()),
                usv[2]->ld,
                raw_pointer_cast(sVector.data()),
                usv[2]->ld,
                &beta,
                raw_pointer_cast(tempMatMult.data()),
                usv[2]->ld);

            cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                usv[2]->m,
                usv[0]->m,
                usv[0]->m,
                &alfa,
                raw_pointer_cast(tempMatMult.data()),
                usv[2]->ld,
                raw_pointer_cast(usv[0]->deviceVector->data()),
                usv[0]->ld,
                &beta,
                raw_pointer_cast(pseudoInverse.data()),
                usv[2]->ld);

            device_vector<float> weightList(chosenAtomIdxList.size());

            cublasSgemv(handle,
                            CUBLAS_OP_N,
                            chosenAtomIdxList.size(),
                            dictionaryMatrix->m,
                            &alfa,
                            raw_pointer_cast(pseudoInverse.data()),
                            chosenAtomIdxList.size(),
                            raw_pointer_cast(patchesMatrix->deviceVector->data()) + (inputIdx * patchesMatrix->m),
                            1,
                            &beta,
                            raw_pointer_cast(weightList.data()),
                            1);            

            //store coefficient 
            transform(weightList.begin(),
                      weightList.end(),
                      make_permutation_iterator(thisSparseCode.begin(),chosenAtomIdxList.begin()),
                      _1);

                cublasSgemv(handle,
                            CUBLAS_OP_N,
                            dictionaryMatrix->m,
                            chosenAtomIdxList.size(),
                            &alfa,
                            raw_pointer_cast(chosenAtomList.data()),
                            dictionaryMatrix->m,
                            raw_pointer_cast(weightList.data()),
                            1,
                            &beta,
                            raw_pointer_cast(tempVec.data()),
                            1);

            transform(patchesMatrix->deviceVector->begin() + (inputIdx * patchesMatrix->m),
                      patchesMatrix->deviceVector->begin() + ((inputIdx+1) * patchesMatrix->m),
                      tempVec.begin(),
                      residualVec.begin(),
                      minus<float>());
            
            norm = 0;
            cublasSnrm2(handle,
                    dictionaryMatrix->m,
                    raw_pointer_cast(residualVec.data()),
                    1,
                    &norm);

            delete container; 
            
            if(norm < 0.001) break;
            
            iter++;
        }
        sparseCode->insert(sparseCode->begin() + (inputIdx * dictionaryMatrix->n), thisSparseCode.begin(), thisSparseCode.end());
    }
    
    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();
    return c;
}


baseUtl::Matrix* CuBlasMatrixOmp::work2(Matrix* dictionaryMatrix, Matrix* patchesMatrix){

    this->b = patchesMatrix;
    this->a = dictionaryMatrix;

    init();

    auto start = std::chrono::steady_clock::now();

    device_vector<float> residualThrust(a->n * b->n);
    float* noisePatches = raw_pointer_cast(a->deviceVector->data());
    float* dictionary = raw_pointer_cast(b->deviceVector->data());
    float* residual = raw_pointer_cast(residualThrust.data());
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
                    noisePatches + (i * dictionaryMatrix->m),
                    1,
                    &beta,
                    residual + (i * dictionaryMatrix->n),
                    1);
    }

    for(int i = 0; i < patchesMatrix->n; i++){ //n == #columns == # patches

        cublasSnrm2(handle,
                    dictionaryMatrix->n,
                    residual + (i * dictionaryMatrix->n),
                    1,
                    &normi);

        int iter = 0;
        iterEpsilon = sqrtf( 1e-4 * normi);
        normf = normi;
        while (normf > iterEpsilon && iter < maxIters){
            
          cublasSgemv(handle,
                    CUBLAS_OP_N,
                    dictionaryMatrix->m,
                    dictionaryMatrix->n,
                    &alfa,
                    dictionary,
                    dictionaryMatrix->ld,
                    residual + (i * dictionaryMatrix->n),
                    1,
                    &beta,
                    noisePatches + (i * dictionaryMatrix->m),
                    1);

            cublasIsamax(handle,
                        dictionaryMatrix->m,
                        noisePatches + (i * dictionaryMatrix->m),
                        1,
                        &q);

            q-=1;

            cublasGetVector(1, 
                            sizeof(coff),
                            noisePatches + q + (i * dictionaryMatrix->m),
                            1,
                            &coff,
                            1);
            
            coff = - coff;

            cublasSaxpy(handle,
                        dictionaryMatrix->n,
                        &coff,
                        dictionary + q,
                        dictionaryMatrix->m,
                        residual + (i * dictionaryMatrix->n),
                        1);

            cublasSnrm2(handle,
                        dictionaryMatrix->n,
                        residual + (i * dictionaryMatrix->n),
                        1,
                        &normf);
            
            iter++;
        }
        
    }
    

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();

    return c;
}