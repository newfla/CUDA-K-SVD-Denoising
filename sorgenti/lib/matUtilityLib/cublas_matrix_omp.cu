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

    CuBlasMatrixOps::init();   

    cVector = new device_vector<float>(b->n * a->n,0);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//******************************
// Clear cuBlas additional data
//*****************************
void CuBlasMatrixOmp::finalize(){

    auto start = std::chrono::steady_clock::now();

    c = new Matrix(a->n, b->n, a->n, cVector);

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

    device_vector<float> residualVec(patchesMatrix->deviceVector->begin(), patchesMatrix->deviceVector->end());
    device_vector<float> tempVec(patchesMatrix->deviceVector->size());
    device_vector<float> proj(dictionaryMatrix->n * patchesMatrix->n);
    host_vector<device_vector<int>> chosenAtomIdxList(patchesMatrix->n);
    host_vector<device_vector<float>> chosenAtomList(patchesMatrix->n);

    float *dictPtr = raw_pointer_cast(dictionaryMatrix->deviceVector->data());
    float *resPtr =  raw_pointer_cast(residualVec.data());
    float *projPtr = raw_pointer_cast(proj.data());
    float *tempPtr = raw_pointer_cast(tempVec.data());
    float *patsPtr = raw_pointer_cast(patchesMatrix->deviceVector->data());

    for(int inputIdx = 0; inputIdx < patchesMatrix->n; inputIdx++){ //n == #columns == # patches        
        
        int iter = 0;
        
        while(iter < maxIters){

            cublasSgemv(*handle,
                        CUBLAS_OP_T,
                        dictionaryMatrix->m,
                        dictionaryMatrix->n,
                        &alfa,
                        dictPtr,
                        dictionaryMatrix->ld,
                        resPtr + (inputIdx * patchesMatrix->m),
                        1,
                        &beta,
                        projPtr + (inputIdx * dictionaryMatrix->n),
                        1);

            cublasIsamax(*handle,
                         dictionaryMatrix->n,
                         projPtr + (inputIdx * dictionaryMatrix->n),
                         1,
                         &max);
            max--;

            cublasIsamin(*handle,
                         dictionaryMatrix->n,
                         projPtr + (inputIdx * dictionaryMatrix->n),
                         1,
                         &min);
            min--;
            
            chosenAtomIdx = (abs(proj[(inputIdx * dictionaryMatrix->n) + max]) > abs(proj[(inputIdx * dictionaryMatrix->n) + min])) ? max : min;

          //  std::cout<<"InputIdx"<<inputIdx<<"IDx scelto: "<<chosenAtomIdx<<"Val:"<<proj[chosenAtomIdx]<<std::endl;
           // std::cin.get();
          
            chosenAtomIdxList[inputIdx].push_back(chosenAtomIdx);

            chosenAtomList[inputIdx].insert(chosenAtomList[inputIdx].end(),
                                  dictionaryMatrix->deviceVector->begin() + (chosenAtomIdx * dictionaryMatrix->m),
                                  dictionaryMatrix->deviceVector->begin() + ((chosenAtomIdx + 1) * dictionaryMatrix->m));
            
            //ChosenAtomList Pseudo-Inverse
            int size = chosenAtomIdxList[inputIdx].size();
            device_vector<float>* copiedList = new device_vector<float> (chosenAtomList[inputIdx].begin(), chosenAtomList[inputIdx].end());
            Matrix* toPinvert = new Matrix(dictionaryMatrix->m, size, dictionaryMatrix->m, copiedList);

            SvdContainer* container = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
            container->setMatrix(toPinvert);
            host_vector<Matrix*> usv = container->getDeviceOutputMatrices();

            device_vector<float> sVector(dictionaryMatrix->m * size,0);            
            device_vector<float> tempMatMult(size * dictionaryMatrix->m);
            device_vector<float> pseudoInverse(size * dictionaryMatrix->m);

            host_vector<int> indicesHost(usv[1]->n);

            for (int i = 0; i < usv[1]->n; i++)
                indicesHost[i] = (i * size) + i;
            
            device_vector<int> indicesDevice = indicesHost;
            
            transform_if(usv[1]->deviceVector->begin(),
                         usv[1]->deviceVector->end(),
                         make_permutation_iterator(sVector.begin(),indicesDevice.begin()),
                         1./_1,
                         not_zero());

            cublasSgemm(*handle,
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

            cublasSgemm(*handle,
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

            device_vector<float> weightList(size);

            cublasSgemv(*handle,
                            CUBLAS_OP_N,
                            size,
                            dictionaryMatrix->m,
                            &alfa,
                            raw_pointer_cast(pseudoInverse.data()),
                            size,
                            patsPtr + (inputIdx * patchesMatrix->m),
                            1,
                            &beta,
                            raw_pointer_cast(weightList.data()),
                            1);            

            //store coefficient
            chosenAtomIdxList[inputIdx][iter] += (dictionaryMatrix->n * inputIdx);
            transform(weightList.begin(),
                      weightList.end(),
                      make_permutation_iterator(cVector->begin(),chosenAtomIdxList[inputIdx].begin()),
                      _1);

                cublasSgemv(*handle,
                            CUBLAS_OP_N,
                            dictionaryMatrix->m,
                            size,
                            &alfa,
                            raw_pointer_cast(chosenAtomList[inputIdx].data()),
                            dictionaryMatrix->m,
                            raw_pointer_cast(weightList.data()),
                            1,
                            &beta,
                            tempPtr + (inputIdx * dictionaryMatrix->m),
                            1);

            transform(patchesMatrix->deviceVector->begin() + (inputIdx * patchesMatrix->m),
                      patchesMatrix->deviceVector->begin() + ((inputIdx+1) * patchesMatrix->m),
                      tempVec.begin() + (inputIdx * dictionaryMatrix->m),
                      residualVec.begin() + (inputIdx * patchesMatrix->m),
                      minus<float>());
            
            norm = 0;
            cublasSnrm2(*handle,
                    dictionaryMatrix->m,
                    resPtr + (inputIdx * patchesMatrix->m),
                    1,
                    &norm);

            delete container; 
            
            if(norm < 0.001) break;
            
            iter++;
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();
    return c;
}


baseUtl::Matrix* CuBlasMatrixOmp::work2(Matrix* patchesMatrix, Matrix* dictionaryMatrix){

    this->b = patchesMatrix;
    this->a = dictionaryMatrix;
    init();
    
    auto start = std::chrono::steady_clock::now();

    device_vector<float> proj(dictionaryMatrix->n * patchesMatrix->n);
    device_vector<float> residualVec(patchesMatrix->deviceVector->begin(), patchesMatrix->deviceVector->end());
    device_vector<float> tempVec(patchesMatrix->deviceVector->size());
    host_vector<device_vector<int>> chosenAtomIdxList(patchesMatrix->n);
    host_vector<device_vector<float>> chosenAtomList(patchesMatrix->n);

    host_vector<float> norms(patchesMatrix->n, 1);

    float *dictPtr = raw_pointer_cast(dictionaryMatrix->deviceVector->data());
    float *resPtr =  raw_pointer_cast(residualVec.data());
    float *projPtr = raw_pointer_cast(proj.data());
    float *tempPtr = raw_pointer_cast(tempVec.data());
    float *patsPtr = raw_pointer_cast(patchesMatrix->deviceVector->data());

    int max, min, chosenAtomIdx, iter = 0;


    while(iter < maxIters){

        cublasSgemm(*handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    dictionaryMatrix->n,
                    patchesMatrix->n,
                    dictionaryMatrix->m,
                    &alfa,
                    dictPtr,
                    dictionaryMatrix->ld,
                    resPtr,
                    patchesMatrix->ld,
                    &beta,
                    projPtr,
                    dictionaryMatrix->n);
        
        for(int inputIdx = 0; inputIdx < patchesMatrix->n; inputIdx++){ //n == #columns == # patches

            if(norms[inputIdx] < 0.001) continue;

            cublasIsamax(*handle,
                         dictionaryMatrix->n,
                         projPtr + (inputIdx * dictionaryMatrix->n),
                         1,
                         &max);

            cublasIsamin(*handle,
                         dictionaryMatrix->n,
                         projPtr + (inputIdx * dictionaryMatrix->n),
                         1,
                         &min);

            max--;
            min--;

            chosenAtomIdx = (abs(proj[(inputIdx * dictionaryMatrix->n) + max]) > abs(proj[(inputIdx * dictionaryMatrix->n) + min])) ? max : min;

            chosenAtomIdxList[inputIdx].push_back(chosenAtomIdx);


            chosenAtomList[inputIdx].insert(chosenAtomList[inputIdx].end(),
                                  dictionaryMatrix->deviceVector->begin() + (chosenAtomIdx * dictionaryMatrix->m),
                                  dictionaryMatrix->deviceVector->begin() + ((chosenAtomIdx + 1) * dictionaryMatrix->m));


            //ChosenAtomList Pseudo-Inverse
            int size = chosenAtomIdxList[inputIdx].size();
            device_vector<float>* copiedList = new device_vector<float> (chosenAtomList[inputIdx].begin(), chosenAtomList[inputIdx].end());
            Matrix* toPinvert = new Matrix(dictionaryMatrix->m, size, dictionaryMatrix->m, copiedList);

            SvdContainer* container = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
            container->setMatrix(toPinvert);
            host_vector<Matrix*> usv = container->getDeviceOutputMatrices();

            device_vector<float> sVector(dictionaryMatrix->m * size,0);            
            device_vector<float> tempMatMult(size * dictionaryMatrix->m);
            device_vector<float> pseudoInverse(size * dictionaryMatrix->m);

            host_vector<int> indicesHost(usv[1]->n);

            for (int i = 0; i < usv[1]->n; i++)
                indicesHost[i] = (i * size) + i;
            
            device_vector<int> indicesDevice = indicesHost;
            
            transform_if(usv[1]->deviceVector->begin(),
                         usv[1]->deviceVector->end(),
                         make_permutation_iterator(sVector.begin(),indicesDevice.begin()),
                         1./_1,
                         not_zero());

            cublasSgemm(*handle,
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


            cublasSgemm(*handle,
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

            delete container;

            device_vector<float> weightList(size);

            cublasSgemv(*handle,
                        CUBLAS_OP_N,
                        size,
                        dictionaryMatrix->m,
                        &alfa,
                        raw_pointer_cast(pseudoInverse.data()),
                        size,
                        patsPtr + (inputIdx * patchesMatrix->m),
                        1,
                        &beta,
                        raw_pointer_cast(weightList.data()),
                        1); 

            //Store coeffs            
            chosenAtomIdxList[inputIdx][iter] += (dictionaryMatrix->n * inputIdx);

           /* if(inputIdx == 256){
                size_t free_byte ;
                size_t total_byte ;
                cudaMemGetInfo( &free_byte, &total_byte ) ;
                std::cout<<"mem free: "<<free_byte<<std::endl;
            }*/

            transform(weightList.begin(),
                      weightList.end(),
                      make_permutation_iterator(cVector->begin(),chosenAtomIdxList[inputIdx].begin()),
                      _1);

            cublasSgemv(*handle,
                        CUBLAS_OP_N,
                        dictionaryMatrix->m,
                        size,
                        &alfa,
                        raw_pointer_cast(chosenAtomList[inputIdx].data()),
                        dictionaryMatrix->m,
                        raw_pointer_cast(weightList.data()),
                        1,
                        &beta,
                        tempPtr + (inputIdx * dictionaryMatrix->m),
                        1);
        }

        transform(patchesMatrix->deviceVector->begin(),
                  patchesMatrix->deviceVector->end(),
                  tempVec.begin(),
                  residualVec.begin(),
                  minus<float>());

        for(int inputIdx = 0; inputIdx < patchesMatrix->n; inputIdx++){ //n == #columns == # patches

            if(norms[inputIdx] < 0.001) continue;
        
            cublasSnrm2(*handle,
                        dictionaryMatrix->m,
                        resPtr + (inputIdx * patchesMatrix->m),
                        1,
                        &norms[inputIdx]);
        }
        iter++;
    }
    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();
    return c;

}