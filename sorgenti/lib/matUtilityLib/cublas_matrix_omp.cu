#include <matUtilityLib.h>
#include <thrust/async/transform.h>

using namespace baseUtl;
using namespace svd;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;

__global__ void copyData1Kernel(device_ptr<int> indicesIterDevice, device_ptr<int> maxs, device_ptr<int> chosenAtomIdxList, device_ptr<int> chosenAtomIdxList2, int dictN, int iter, int iters, int tot, int stride){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;
    int inputIdx = indicesIterDevice[tid];
    int chosenAtomIdx = maxs[inputIdx];
    int inputIdxStride = inputIdx + stride;
    chosenAtomIdxList[(inputIdxStride * iters) + iter] = (dictN * inputIdxStride) + chosenAtomIdx;
    chosenAtomIdxList2[(inputIdxStride * iters) + iter] = chosenAtomIdx;
    
  //  for(int i=0; i<dictM; i++)
   //     chosenAtomList[(inputIdxStride * iters * dictM) + (iter * dictM) + i] = dict[(chosenAtomIdx * dictM) + i];
}

__global__ void copyData2Kernel(device_ptr<int> indicesIterDevice,  device_ptr<int> chosenAtomIdxList2, device_ptr<float> dict, device_ptr<float> copiedList, int dictM, int iter, int iters, int tot, int stride){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;
    int inputIdx = indicesIterDevice[tid];
    int inputIdxStride = inputIdx + stride;

    for(int i = 0; i < iter; i++){
        int chosenAtomIdx = chosenAtomIdxList2[(inputIdxStride * iters) + i];
        for(int j = 0; j < dictM; j++){
            copiedList[(dictM * tid * iter) + (i * dictM) + j] = dict[(dictM * chosenAtomIdx) + j];
        }
    }

       // copiedList[(dictM * tid * iter) + i] =  chosenAtomList[((inputIdx + stride) * iters * dictM) + i];
}

__global__ void copyData3Kernel(device_ptr<float> weightList, device_ptr<float> cVector, device_ptr<int> indicesIterDevice, device_ptr<int> chosenAtomIdxList, int size, int iters, int tot, int stride){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;
    int inputIdx = indicesIterDevice[tid];

    for (int i = 0; i < size; i++){
        int idx = chosenAtomIdxList[((inputIdx + stride) * iters) + i]; 
        cVector[idx] = weightList[(size * inputIdx) + i];
    }
    
}

__global__ void transformSMat(device_ptr<float> s, device_ptr<float> sVector, int n, int m, int tot){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;
    for (int i = 0; i < n; i++) 
    {
        float x = s[(n * tid) + i];
        if(x != 0)
            sVector[(m * n * tid) + (i * n) + i ] = 1./x;
    }
    
}


CuBlasMatrixOmp::CuBlasMatrixOmp(){}

//********************************************************
// Create cuBlas additional data and move data on DEVICE
//*******************************************************
void CuBlasMatrixOmp::init(){

    auto start = std::chrono::steady_clock::now();

    CuBlasMatrixOps::init();   

    if(cVector != NULL){
        cVector->clear();
        cVector->shrink_to_fit();
        delete cVector;
    }
    cVector = new device_vector<float>(b->n * a->n,0);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//******************************
// Clear cuBlas additional data
//*****************************
void CuBlasMatrixOmp::finalize(){

    auto start = std::chrono::steady_clock::now();

    cublasSetPointerMode(*handle, CUBLAS_POINTER_MODE_HOST);
    c = new Matrix(a->n, b->n, a->n, cVector);

    auto end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

thrust::host_vector<cudaStream_t>* matUtl::CuBlasMatrixOmp::streams = NULL;

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

    int iter = 0;
    device_vector<float> normsDevice(patchesMatrix->n, 1);
    host_vector<float> norms(patchesMatrix->n, 1);
    device_vector<float> residualVec(patchesMatrix->deviceVector->begin(), patchesMatrix->deviceVector->end());

    if(proj == NULL){
        blocks = patchesMatrix->n / 1024;
        blocks += (patchesMatrix->n % 1024 > 0) ? 1 : 0;

        alfaBeta = new device_vector<float>(2,1);
        chosenAtomIdxList = new device_vector<int>(patchesMatrix->n * maxIters);
        chosenAtomIdxList2 = new device_vector<int>(patchesMatrix->n * maxIters);
        tempVec = new device_vector<float>(patchesMatrix->deviceVector->size());
        alfaBeta->data()[1] = 0;

        cudaMemGetInfo(&free_byte, &total_byte);

        size_t spaceRequired = 2 * sizeof(float) * dictionaryMatrix->n * patchesMatrix->n; //proj + projAbs
        spaceRequired += 2 * sizeof(int) * patchesMatrix->n; //maxs + indicesIterDevice
        spaceRequired += 2 * sizeof(float) * patchesMatrix->n * maxIters * dictionaryMatrix->m; //tempMatMult + pseudoInverse
        spaceRequired += sizeof(float) * patchesMatrix->n * maxIters; //weightList
        spaceRequired += sizeof(float) * patchesMatrix->n; //normsDevice
        spaceRequired += sizeof(float) * patchesMatrix->n * dictionaryMatrix->m * maxIters; // sVector
        spaceRequired += sizeof(float) * patchesMatrix->n * dictionaryMatrix->m * maxIters; //copiedList
        spaceRequired += sizeof(float) * patchesMatrix->n * dictionaryMatrix->m * dictionaryMatrix->m; //U
        spaceRequired += sizeof(float) * patchesMatrix->n * maxIters; //S
        spaceRequired += sizeof(float) * patchesMatrix->n * maxIters * maxIters; //VT

        subIter = spaceRequired / free_byte;
        subIter++;
        subIter += (subIter == 1) ? 0 : minOmpIterBatch;

        int patchesXIter = patchesMatrix->n / subIter;
        patchesXIter += (subIter == 1) ? 0 : 1;

	    if(subIter>1 && patchesMatrix->n % patchesXIter == 0)
	    subIter--;
        patchesIter = new host_vector<int>(subIter, patchesXIter);

        if(subIter > 1 && (patchesMatrix->n % subIter !=0)){
            int temp = reduce(patchesIter->begin(), patchesIter->end() - 1);
            patchesIter->data()[subIter-1] = patchesMatrix->n - temp;
        }

        proj = new device_vector<float> (dictionaryMatrix->n * patchesXIter);
        projAbs = new device_vector<float> (dictionaryMatrix->n * patchesXIter);
        maxs = new device_vector<int>(patchesXIter);
        tempMatMult = new device_vector<float>(patchesXIter * maxIters * dictionaryMatrix->m);
        pseudoInverse = new device_vector<float>(patchesXIter * maxIters * dictionaryMatrix->m);
        weightList = new device_vector<float>(patchesXIter * maxIters);

    }

    float *dictPtr = raw_pointer_cast(dictionaryMatrix->deviceVector->data());
    float *resPtr =  raw_pointer_cast(residualVec.data());
    float *projPtr = raw_pointer_cast(proj->data());
    float *projAbsPtr = raw_pointer_cast(projAbs->data());
    float *tempPtr = raw_pointer_cast(tempVec->data());
    float *patsPtr = raw_pointer_cast(patchesMatrix->deviceVector->data());
    float *normsDevicePtr = raw_pointer_cast(normsDevice.data());
    float *alfaBetaPtr = raw_pointer_cast(alfaBeta->data());
    float *tempMatMultPtr = raw_pointer_cast(tempMatMult->data());
    float *pseudoInversePtr = raw_pointer_cast(pseudoInverse->data());
    float *weightListPtr = raw_pointer_cast(weightList->data());
    int *maxsPtr = raw_pointer_cast(maxs->data());
    
     if(streams == NULL){
        streams = new host_vector<cudaStream_t>(maxStreams);
        for(int inputIdx = 0; inputIdx < maxStreams; inputIdx++)
            cudaStreamCreate(&(streams->data()[inputIdx]));
    }
   
    while(iter < maxIters){
        int countPatches = 0;
        host_vector<int>indicesIterGlobal;
        int ppp = 0;
        for(int maxPatches : *patchesIter){
            ppp++;
     	    cublasSetPointerMode(*handle, CUBLAS_POINTER_MODE_HOST);       

            cublasSgemm(*handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        dictionaryMatrix->n,
                        maxPatches,
                        dictionaryMatrix->m,
                        &alfa,
                        dictPtr,
                        dictionaryMatrix->ld,
                        resPtr + (countPatches * dictionaryMatrix->m),
                        patchesMatrix->ld,
                        &beta,
                        projPtr,
                        dictionaryMatrix->n);

            cudaDeviceSynchronize();

            transform(proj->begin(), proj->end(), projAbs->begin(), abs_val());

             cublasSetPointerMode(*handle, CUBLAS_POINTER_MODE_DEVICE);

            host_vector<int>indicesIter;

            for(int inputIdx = 0; inputIdx < maxPatches; inputIdx++){

                if(norms[inputIdx + countPatches] < 0.001) continue;

                indicesIter.push_back(inputIdx);
                indicesIterGlobal.push_back(inputIdx + countPatches);

                cublasSetStream(*handle,streams->data()[inputIdx % maxStreams]);

                cublasIsamax(*handle,
                            dictionaryMatrix->n,
                            projAbsPtr + (inputIdx  * dictionaryMatrix->n),
                            1,
                            maxsPtr + inputIdx);
            }

            cudaDeviceSynchronize();
            transform(maxs->begin() , maxs->end(), maxs->begin()  , _1 - 1);

            int size = iter + 1;
            int max = size * indicesIter.size();
            if(max == 0){
                countPatches += maxPatches;
                continue;
            }
 
            device_vector<int> indicesIterDevice = indicesIter;
            copyData1Kernel<<<blocks, 1024>>> (indicesIterDevice.data(), maxs->data(),  chosenAtomIdxList->data(), chosenAtomIdxList2->data(), dictionaryMatrix->n, iter, maxIters, indicesIter.size(), countPatches);
            cudaDeviceSynchronize();
  
            device_vector<float>* copiedList = new device_vector<float>(max * dictionaryMatrix->m);
            device_vector<float> sVector(indicesIter.size() * dictionaryMatrix->m * size,0);            
            float *sVectorPtr = raw_pointer_cast(sVector.data());

            copyData2Kernel<<<blocks, 1024>>>(indicesIterDevice.data(), chosenAtomIdxList2->data(), dictionaryMatrix->deviceVector->data(), copiedList->data(), dictionaryMatrix->m, size, maxIters, indicesIter.size(), countPatches);
            cudaDeviceSynchronize();
 
            //ChosenAtomList Pseudo-Inverse
            Matrix* toPinvert = new Matrix(dictionaryMatrix->m, size, indicesIter.size(), copiedList);
            SvdContainer* container = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDA_BATCH)); 
            container->setMatrix(toPinvert);
            host_vector<Matrix*> usv = container->getDeviceOutputMatrices();
            transformSMat<<<blocks, 1024>>>(usv[1]->deviceVector->data(), sVector.data(), size, dictionaryMatrix->m, indicesIter.size());
            
            cudaDeviceSynchronize();
            cublasSetPointerMode(*handle, CUBLAS_POINTER_MODE_HOST);

            cublasSgemmStridedBatched(*handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    usv[2]->m,
                                    usv[0]->m,
                                    usv[2]->n,
                                    &alfa,
                                    raw_pointer_cast(usv[2]->deviceVector->data()),
                                    usv[2]->m,
                                    usv[2]->ld,
                                    sVectorPtr,
                                    usv[2]->m,
                                    dictionaryMatrix->m * size,
                                    &beta,
                                    tempMatMultPtr,
                                    usv[2]->m,
                                    dictionaryMatrix->m * size,
                                    indicesIter.size());

            cudaDeviceSynchronize();

            cublasSgemmStridedBatched(*handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_T,
                                    usv[2]->m,
                                    usv[0]->m,
                                    usv[0]->m,
                                    &alfa,
                                    tempMatMultPtr,
                                    usv[2]->m,
                                    dictionaryMatrix->m * size,
                                    raw_pointer_cast(usv[0]->deviceVector->data()),
                                    usv[0]->m,
                                    usv[0]->ld,
                                    &beta,
                                    pseudoInversePtr,
                                    usv[2]->m,
                                    dictionaryMatrix->m * size,
                                    indicesIter.size());
            cudaDeviceSynchronize();
             
            int count = -1;
            cublasSetPointerMode(*handle, CUBLAS_POINTER_MODE_DEVICE);
            for(int inputIdx: indicesIter){
                
                count++;
                cublasSetStream(*handle,streams->data()[inputIdx % maxStreams]);

                cublasSgemv(*handle,
                            CUBLAS_OP_N,
                            size,
                            dictionaryMatrix->m,
                            alfaBetaPtr,
                            pseudoInversePtr + (dictionaryMatrix->m * size * count),
                            size,
                            patsPtr + ((inputIdx + countPatches) * patchesMatrix->m),
                            1,
                            alfaBetaPtr + 1,
                            weightListPtr + (size * inputIdx),
                            1);
            }
 
            cudaDeviceSynchronize();
            copyData3Kernel<<<blocks,1024>>>(weightList->data(), cVector->data(), indicesIterDevice.data(),  chosenAtomIdxList->data(), size, maxIters, indicesIter.size(), countPatches);
            cudaDeviceSynchronize();
 
            //cudaMemGetInfo( &free_byte, &total_byte ) ;
           // std::cout<<"   mem freeAfter: "<<free_byte/1073741824.<<std::endl;

            int inputIdxCounter = 0;
            for(int inputIdx: indicesIter){
                
                cublasSetStream(*handle,streams->data()[inputIdx % maxStreams]);

                cublasSgemv(*handle,
                            CUBLAS_OP_N,
                            dictionaryMatrix->m,
                            size,
                            alfaBetaPtr,
                            raw_pointer_cast(copiedList->data()) + (inputIdxCounter * dictionaryMatrix->m * size),
                            dictionaryMatrix->m,
                            weightListPtr + (size * inputIdx),
                            1,
                            alfaBetaPtr + 1,
                            tempPtr + ((inputIdx + countPatches) * dictionaryMatrix->m),
                            1); 
                inputIdxCounter++;
            }        
            cudaDeviceSynchronize();
            delete container; 
            countPatches += maxPatches;
        }    
    
        transform(patchesMatrix->deviceVector->begin(),
                  patchesMatrix->deviceVector->end(),
                  tempVec->begin(),
                  residualVec.begin(),
                  minus<float>());

            for(int inputIdx : indicesIterGlobal){ //n == #columns == # patches

                cublasSetStream(*handle,streams->data()[inputIdx % maxStreams]);
            
                cublasSnrm2(*handle,
                            dictionaryMatrix->m,
                            resPtr + (inputIdx * patchesMatrix->m),
                            1,
                            normsDevicePtr + inputIdx);
            }
            cudaDeviceSynchronize();
            norms = normsDevice;
            iter++;
    }

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();
    //cudaMemGetInfo( &free_byte, &total_byte );
    //std::cout<<"\nmem free5: "<<free_byte/1073741824.<<std::endl;
    return c;
}