#include <matUtilityLib.h>
#include <thrust/async/transform.h>

using namespace baseUtl;
using namespace svd;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;

__global__ void copyData1Kernel(device_ptr<int> indicesIterDevice, device_ptr<int> maxs, device_ptr<int> chosenAtomIdxList, device_ptr<float> chosenAtomList, device_ptr<float> dict, int dictM, int dictN, int iter, int iters, int tot){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;
    int inputIdx = indicesIterDevice[tid];
    int chosenAtomIdx = maxs[inputIdx];
    
    chosenAtomIdxList[(inputIdx * iters) + iter] = (dictN * inputIdx) + chosenAtomIdx;
    
    for(int i=0; i<dictM; i++)
        chosenAtomList[(inputIdx * iters * dictM) + (iter * dictM) + i] = dict[(chosenAtomIdx * dictM) + i];
}

__global__ void copyData2Kernel(device_ptr<int> indicesIterDevice, device_ptr<float> chosenAtomList, device_ptr<float> copiedList, int dictM, int iter, int iters, int tot){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;
    int inputIdx = indicesIterDevice[tid];
    for(int i=0; i < iter * dictM; i++)
        copiedList[(dictM * tid * iter) + i] =  chosenAtomList[(inputIdx * iters * dictM) + i];
}

__global__ void copyData3Kernel(device_ptr<float> weightList, device_ptr<float> cVector, device_ptr<int> indicesIterDevice, device_ptr<int> chosenAtomIdxList, int size, int iters, int tot){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;
    int inputIdx = indicesIterDevice[tid];

    for (int i = 0; i < size; i++){
        int idx = chosenAtomIdxList[(inputIdx * iters) + i]; 
        cVector[idx] = weightList[(size * inputIdx) + i];
    }
    
}

__global__ void transformSMat(device_ptr<float> s, device_ptr<float> sVector, int n, int m, int tot)
{
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
    if(proj == NULL){
        proj = new device_vector<float> (dictionaryMatrix->n * patchesMatrix->n);
        projAbs = new device_vector<float> (dictionaryMatrix->n * patchesMatrix->n);
        tempVec = new device_vector<float>(patchesMatrix->deviceVector->size());
        maxs = new device_vector<int>(patchesMatrix->n);
        tempMatMult = new device_vector<float>(patchesMatrix->n * maxIters * dictionaryMatrix->m);
        pseudoInverse = new device_vector<float>(patchesMatrix->n * maxIters * dictionaryMatrix->m);
        weightList = new device_vector<float>(patchesMatrix->n * maxIters);
        alfaBeta = new device_vector<float>(2,1);
        chosenAtomIdxList = new device_vector<int>(patchesMatrix->n * maxIters);
        chosenAtomList = new device_vector<float>(patchesMatrix->n * maxIters * dictionaryMatrix->m);

        alfaBeta->data()[1] = 0;
        blocks = patchesMatrix->n / 1024;
        blocks += (patchesMatrix->n % 1024 > 0) ? 1 : 0;
    }

    device_vector<float> residualVec(patchesMatrix->deviceVector->begin(), patchesMatrix->deviceVector->end());
    device_vector<float>normsDevice(patchesMatrix->n, 1);
    host_vector<float>norms(patchesMatrix->n, 1);

    float *dictPtr = raw_pointer_cast(dictionaryMatrix->deviceVector->data());
    float *resPtr =  raw_pointer_cast(residualVec.data());
    float *projPtr = raw_pointer_cast(proj->data());
    float *projAbsPtr = raw_pointer_cast(projAbs->data());
    float *tempPtr = raw_pointer_cast(tempVec->data());
    float *patsPtr = raw_pointer_cast(patchesMatrix->deviceVector->data());
    float *normsDevicePtr = raw_pointer_cast(normsDevice.data());
    float* alfaBetaPtr = raw_pointer_cast(alfaBeta->data());
    int *maxsPtr = raw_pointer_cast(maxs->data());
    float *tempMatMultPtr = raw_pointer_cast(tempMatMult->data());
    float *pseudoInversePtr = raw_pointer_cast(pseudoInverse->data());
    float *weightListPtr = raw_pointer_cast(weightList->data());

    int iter = 0;
    size_t free_byte, total_byte;
    
    
     if(streams == NULL){
        streams = new host_vector<cudaStream_t>(maxStreams);
        for(int inputIdx = 0; inputIdx < maxStreams; inputIdx++)
            cudaStreamCreate(&(streams->data()[inputIdx]));
    }
   
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

        cudaDeviceSynchronize();

        transform(proj->begin(), proj->end(), projAbs->begin(), abs_val());

        cublasSetPointerMode(*handle, CUBLAS_POINTER_MODE_DEVICE);

        host_vector<int>indicesIter;

        for(int inputIdx = 0; inputIdx < patchesMatrix->n; inputIdx++){

            if(norms[inputIdx] < 0.001) continue;

            indicesIter.push_back(inputIdx);

            cublasSetStream(*handle,streams->data()[inputIdx % maxStreams]);

            cublasIsamax(*handle,
                         dictionaryMatrix->n,
                         projAbsPtr + (inputIdx * dictionaryMatrix->n),
                         1,
                         maxsPtr + inputIdx);
        }

        cudaDeviceSynchronize();
        transform(maxs->begin() , maxs->end(), maxs->begin()  , _1-1);

        int size = iter + 1;

        device_vector<int> indicesIterDevice = indicesIter;
        copyData1Kernel<<<blocks, 1024>>> (indicesIterDevice.data(), maxs->data(),  chosenAtomIdxList->data(),  chosenAtomList->data(), dictionaryMatrix->deviceVector->data(), dictionaryMatrix->m, dictionaryMatrix->n, iter, maxIters, indicesIter.size());
        cudaDeviceSynchronize();
        int max = (iter + 1) * indicesIter.size();

        device_vector<float>* copiedList = new device_vector<float>(max * dictionaryMatrix->m);
        device_vector<float> sVector(indicesIter.size() * dictionaryMatrix->m * size,0);            
        float *sVectorPtr = raw_pointer_cast(sVector.data());

        copyData2Kernel<<<blocks, 1024>>>(indicesIterDevice.data(),  chosenAtomList->data(), copiedList->data(), dictionaryMatrix->m, size, maxIters, indicesIter.size());
        cudaDeviceSynchronize();

        //ChosenAtomList Pseudo-Inverse
        Matrix* toPinvert = new Matrix(dictionaryMatrix->m, size, indicesIter.size(), copiedList);
        SvdContainer* container = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDA_BATCH)); 
        container->setMatrix(toPinvert);
        host_vector<Matrix*> usv = container->getDeviceOutputMatrices();
        
        cudaMemGetInfo( &free_byte, &total_byte ) ;
        //std::cout<<"mem free5: "<<free_byte/1073741824.<<std::endl;

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

        delete container;
        
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
                        patsPtr + (inputIdx * patchesMatrix->m),
                        1,
                        alfaBetaPtr + 1,
                        weightListPtr + (size * inputIdx),
                        1);
        }

        cudaDeviceSynchronize();
        copyData3Kernel<<<blocks,1024>>>(weightList->data(), cVector->data(), indicesIterDevice.data(),  chosenAtomIdxList->data(), size, maxIters, indicesIter.size());
        cudaDeviceSynchronize();
        
        for(int inputIdx: indicesIter){
            
            cudaStreamSynchronize(streams->data()[inputIdx % maxStreams]);
            
            cublasSetStream(*handle,streams->data()[inputIdx % maxStreams]);

            cublasSgemv(*handle,
                        CUBLAS_OP_N,
                        dictionaryMatrix->m,
                        size,
                        alfaBetaPtr,
                        raw_pointer_cast(chosenAtomList->data()) + (inputIdx * dictionaryMatrix->m * maxIters),
                        dictionaryMatrix->m,
                        weightListPtr + (size * inputIdx),
                        1,
                        alfaBetaPtr + 1,
                        tempPtr + (inputIdx * dictionaryMatrix->m),
                        1); 
        }        

        cudaDeviceSynchronize();

        transform(patchesMatrix->deviceVector->begin(),
                  patchesMatrix->deviceVector->end(),
                  tempVec->begin(),
                  residualVec.begin(),
                  minus<float>());
       // std::cout<<"----------------------------------\n\n";

        for(int inputIdx : indicesIter){ //n == #columns == # patches

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
        cublasSetPointerMode(*handle, CUBLAS_POINTER_MODE_HOST);
    }
    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    finalize();
    return c;

}