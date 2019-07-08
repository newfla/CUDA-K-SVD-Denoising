#include <denoisingLib.h>

using namespace denoising;
using namespace baseUtl;
using namespace svd;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;
using namespace cimg_library;

__global__ void computeErrorKernel(device_ptr<float> patches, device_ptr<float> dx, device_ptr<int> relevantDataIndices, int offsetIndices, int dim, int tot){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;

    int start = relevantDataIndices[offsetIndices + tid] * dim;

    for (int i = 0; i < dim; i++)
        dx[(tid * dim) + i] = patches[start + i] - dx[(tid * dim) + i];
}


__global__ void copyTransformSparseCodeKernel(device_ptr<float> sparseCode, device_ptr<float> selectSparseCode, device_ptr<int> relevantDataIndices, int offsetIndices, int atom, int atoms, int tot){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;

    int start = relevantDataIndices[offsetIndices + tid];

    copy(thrust::device, 
         sparseCode + (start * atoms),
         sparseCode + ((start + 1) * atoms),
         selectSparseCode  + (tid * atoms));

    selectSparseCode[(tid * atoms) + atom] = 0.; 
}

__global__ void findRelevantIndicesKernel(device_ptr<float> sparseCode, device_ptr<int> relevantDataIndices, device_ptr<int> relevantDataIndicesCounter,int patches, int atoms){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=atoms)
        return;

    int start = tid * patches;
    int counter = 0;
    for (int i = 0; i < patches; i++){
        if(sparseCode[(i * atoms) + tid] != 0){
            relevantDataIndices[start + counter] = i;
            counter++;
        }
    }
    relevantDataIndicesCounter[tid] = counter;
}

__global__ void normalizeDictKernel(device_ptr<float> dict, int n, int tot){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid>=tot)
        return;
    int start = tid * n, end = start + n;
    //Calculate norm
    float norm = sqrtf(transform_reduce(thrust::device, dict + start, dict + end, square<float>(), 0, plus<float>()));
    
    //Normalize vector
    transform(thrust::device, dict + start , dict + end, dict + start, _1/norm);
}

CudaKSvdDenoiser::CudaKSvdDenoiser(){}

CudaKSvdDenoiser::~CudaKSvdDenoiser(){}

//**************************
//  Istantiate SvdContainer
//*************************
SvdContainer* CudaKSvdDenoiser::buildSvdContainer(){
    
    switch (type)
        {
            case CUDA_K_GESVD:
                return new SvdContainer(SvdEngine::factory(CUSOLVER_GESVD));
            case CUDA_K_GESVDA:
                return new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDA_BATCH)); 
            default:
            case CUDA_K_GESVDJ:
                return new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
         
        }
}

//***********************************************************************************************************************
//  Load denoising save
//  output:  + status (signed char) 0 = done, -1 = image loading failed, -2 = denoising failed, -3 = image saving failed
//**********************************************************************************************************************
signed char CudaKSvdDenoiser::denoising(){

    if(!loadImage())
        return -1;    
    
    if(!internalDenoising())
        return -2;

    if(!saveImage())
        return -3;

    return 0;
}

//**************************
//  Load image
//  output:  + staus (bool)
//*************************
bool CudaKSvdDenoiser::loadImage(){

    bool a = Denoiser::loadImage();

    if(speckle){
        transform(inputMatrix->hostVector->begin(),inputMatrix->hostVector->end(),inputMatrix->hostVector->begin(),myLog<float>());
        CImg<float>* image = new CImg<float>(inputMatrix->m, inputMatrix->n);   
        image->_data = inputMatrix->hostVector->data();
        sigma = (float)image->variance_noise();
    }
    return a;
}

//**************************
//  Save image
//  output:  + staus (bool)
//*************************
bool CudaKSvdDenoiser::saveImage(){

    return Denoiser::saveImage();
}

//****************************
//  CUDA K-SVD implementation 
//  output:  + staus (bool)host
//******************************
bool CudaKSvdDenoiser::internalDenoising(){

    auto start = std::chrono::steady_clock::now();

    if(subImageWidthDim != 0 && subImageHeightDim != 0){
        int strideHeight = subImageHeightDim;
        int strideWidth = subImageWidthDim;
        std::swap(subImageHeightDim, patchHeightDim);
        std::swap(subImageWidthDim, patchWidthDim);
        std::swap(slidingHeight, strideHeight);
        std::swap(slidingWidth, strideWidth);

        createPatches(false);
        Matrix* imagePatches = noisePatches;
        Matrix* inputMatrixPointer = inputMatrix;

        std::swap(subImageHeightDim, patchHeightDim);
        std::swap(subImageWidthDim, patchWidthDim);
        std::swap(slidingHeight, strideHeight);
        std::swap(slidingWidth, strideWidth);
        CImg<float>* tempImage = new CImg<float>(subImageHeightDim, subImageWidthDim);

        for(int i = 0; i < imagePatches->n; i++){
            
            std::cout<<"subImage: "<<i+1<<" / "<<imagePatches->n<<std::endl;

            //Preapare SubImage
            host_vector<float>* subImage = new host_vector<float>(imagePatches->hostVector->begin() + (imagePatches->m * i), imagePatches->hostVector->begin() + (imagePatches->m * (i+1)));
           
            inputMatrix = new Matrix(subImageHeightDim, subImageWidthDim, subImageHeightDim, subImage->data()); 
              
            tempImage->_data = subImage->data();
            sigma = tempImage->variance_noise();
            
            //Divide image in patches column majhostor of fixed dims
            createPatches(true);
            
            //Init Dict
            initDictionary();

            //Start #iter K-SVD
            kSvd();

            //Rebuild originalImage
            createImage(false);

            copy(outputMatrix->hostVector->begin(), outputMatrix->hostVector->end(), imagePatches->hostVector->begin() + (imagePatches->m * i));
            delete outputMatrix;
            delete sparseCode;
            delete dictionary;
            delete noisePatches;
            delete inputMatrix;

            //cudaMemGetInfo( &free_byte, &total_byte );
            //std::cout<<"\nmem free5: "<<free_byte/1073741824.<<std::endl;
        }
        createImageFromSubImages(imagePatches, inputMatrixPointer);

        delete inputMatrixPointer;
        delete imagePatches;
        delete tempImage;
    }else{

        //Divide image in patches column majhostor of fixed dims
        createPatches(true);

        //Init Dict
        initDictionary();

        //Start #iter K-SVD
        kSvd();

        //Rebuild originalImage
        createImage(true);

        delete sparseCode;
        delete dictionary;
        delete noisePatches;
    }

    

    CuBlasMatrixOps::finalize();
    SvdCudaEngine::finalize();
   
    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return true;
}

//**********************************************************************************************
//  Divide image in square patches column major of fixed dims (patchWidthDim x patchHeightDim)
//*********************************************************************************************
void CudaKSvdDenoiser::createPatches(bool transfer){

   //std::cout<<"Create Patches"<<std::endl;

    auto start = std::chrono::steady_clock::now();

    int i, j;
    host_vector<float>* patchesHost = new host_vector<float>();

    //Create patch division on host
    for(int i = 0; i + patchHeightDim <= inputMatrix->n; i+= slidingHeight){ 

        for(j = 0; j + patchWidthDim <= inputMatrix->m; j+= slidingWidth){ 

            int startPatch = (i * inputMatrix->m) + j;

            for(int k = startPatch; k < startPatch + patchHeightDim * inputMatrix->m; k += inputMatrix->m)
               patchesHost->insert(patchesHost->end(), inputMatrix->hostVector->begin() + k, inputMatrix->hostVector->begin() + k + patchWidthDim);
        }  
    }

    i = patchWidthDim * patchHeightDim;
    j = patchesHost->size() / i;
    noisePatches = new Matrix(i, j, i, patchesHost);    

    if(transfer)
        noisePatches->copyOnDevice();
    
    //std::cout<<"    # Patches: "<<j<<"  Dim: "<<i<<std::endl;

    std::swap(inputMatrix->m, inputMatrix->n);
    inputMatrix->ld = inputMatrix->m;
    
    auto end = std::chrono::steady_clock::now();
    //std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(end-start).count()<<" s"<<std::endl<<std::endl;
}

//*************************************************************************************************************
//  Init a dictionary using #atoms patches column major of fixed dims (patchWidthDim x patchHeightDim)
//************************************************************************************************************
void CudaKSvdDenoiser::initDictionary(){

    //std::cout<<"Init Dictionary"<<std::endl;

    auto start = std::chrono::steady_clock::now();
    int dim = patchWidthDim * patchHeightDim;
    int offset = dim * (noisePatches->n / 2);
    device_vector<float>* dict = new device_vector<float>(noisePatches->hostVector->begin() + offset, noisePatches->hostVector->begin() + offset + (dim * atoms));

    dictionary = new Matrix(dim, atoms, dim, dict);

    int blocks = atoms / 1024;
    blocks += (atoms % 1024 >0) ? 1 : 0;

    //Normalize patches using norm2
    normalizeDictKernel<<<blocks, 1024>>>(dictionary->deviceVector->data(), dim, atoms);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    //std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<" ms"<<std::endl<<std::endl;
}

//******************************************************
//  Update dictionary columns using SVD on Error Matrix
//*****************************************************
void CudaKSvdDenoiser::updateDictionary(){

    CuBlasMatrixMult* mult;
    Matrix* dx;
    Matrix* v;
    Matrix* sparseMatrix;
    host_vector<int> relevantDataIndicesCounterHost;
    device_vector<int> relevantDataIndices (sparseCode->m * sparseCode->n);
    device_vector<int> relevantDataIndicesCounter (sparseCode->m);
    
    blocks = atoms / 1024;
    blocks += (atoms % 1024 >0) ? 1 : 0;

    findRelevantIndicesKernel<<<blocks,1024>>>(sparseCode->deviceVector->data(), relevantDataIndices.data(), relevantDataIndicesCounter.data(), sparseCode->n, sparseCode->m);
    cudaDeviceSynchronize();

    relevantDataIndicesCounterHost = relevantDataIndicesCounter;
    
    
    int totRelevant = reduce(relevantDataIndicesCounterHost.begin(), relevantDataIndicesCounterHost.end(), 0, maximum<int>());
    int offset = 0;

    device_vector<float> selectSparseCode(sparseCode->m * totRelevant);
    sparseMatrix = new Matrix(sparseCode->m, totRelevant, sparseCode->m, &selectSparseCode);

    for(int atomIdx = 0 ; atomIdx < sparseCode->m ; atomIdx++){ //->m = # atoms

        if(relevantDataIndicesCounterHost[atomIdx] < 1)
            continue;
        offset = sparseCode->n * atomIdx;

        copyTransformSparseCodeKernel<<<blocks,1024>>>(sparseCode->deviceVector->data(), selectSparseCode.data(), relevantDataIndices.data(), offset, atomIdx, sparseCode->m, relevantDataIndicesCounterHost[atomIdx]);
        cudaDeviceSynchronize();
      
        //DX = Dictionary * selectSparseCode
        mult = (CuBlasMatrixMult*) MatrixOps::factory(CUBLAS_MULT);
        ((CuBlasMatrixMult*)mult)->setOps(CUBLAS_OP_N, CUBLAS_OP_N);
        sparseMatrix->n = relevantDataIndicesCounterHost[atomIdx];
        dx = mult->work(dictionary, sparseMatrix);
     
        computeErrorKernel<<<blocks,1024>>>(noisePatches->deviceVector->data(), dx->deviceVector->data(), relevantDataIndices.data(), offset, dx->m, dx->n);
        cudaDeviceSynchronize();

        //Compute SVD on E
        SvdContainer* svdContainer = buildSvdContainer();
        svdContainer->setMatrix(dx);
        if(type == CUDA_K_GESVDA)
            dx->ld = 1;
        host_vector<Matrix*> usv = svdContainer->getDeviceOutputMatrices();

        //Traspose V
        if(type == CUDA_K_GESVD){
            MatrixOps* tras = MatrixOps::factory(CUBLAS_ADD);
            ((CuBlasMatrixAdd*)tras)->setOps(CUBLAS_OP_T, CUBLAS_OP_T);
            v = tras->work(usv[2], usv[2]);
        }
        else
            v = usv[2];
        //Replace dictionary column   
        transform(usv[0]->deviceVector->begin(),
                  usv[0]->deviceVector->begin() + usv[0]->m,
                  dictionary->deviceVector->begin() + (atomIdx * dictionary->m),
                  _1 * -1.f);

        //Calculate new coeffs
        device_vector<int> idxSeq(relevantDataIndicesCounterHost[atomIdx]); 
        sequence(idxSeq.begin(), idxSeq.end(), atomIdx, sparseCode->m);
        transform(relevantDataIndices.begin() + offset,
                  relevantDataIndices.begin() + offset + relevantDataIndicesCounterHost[atomIdx],
                  idxSeq.begin(),  atomIdx + (_1 * sparseCode->m));        
       
        transform(v->deviceVector->begin(),
                  v->deviceVector->begin() + relevantDataIndicesCounterHost[atomIdx],
                  make_permutation_iterator(sparseCode->deviceVector->begin(), idxSeq.begin()),
                  -1.f * usv[1]->deviceVector->data()[0] *_1);

        delete svdContainer;
    }
}

//******************
//  kSVD algorithm
//  GPU Version
//*************
void CudaKSvdDenoiser::kSvd(){
    
    CuBlasMatrixOmp* omp = (CuBlasMatrixOmp*) MatrixOps::factory(CUBLAS_OMP);
    omp->maxIters = ompIter;
    omp->minOmpIterBatch = minOmpIterBatch;
    
    for(int i = 0 ; i < iter ; i++){

       // std::cout<<"Ksvd-Iter: "<<i+1<<std::endl;

        //OMP phase
        auto start = std::chrono::steady_clock::now();
     
        sparseCode = omp->work(noisePatches, dictionary);

        auto end = std::chrono::steady_clock::now();
        auto tot1 = end - start;
       // std::cout<<"    # OMP Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(tot1).count()<<" s"<<std::endl;

        //Dict update phase
        start = std::chrono::steady_clock::now();
        updateDictionary();
        end = std::chrono::steady_clock::now();
        auto tot2 = end - start;
     //   std::cout<<"    # Dict update Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(tot2).count()<<" s"<<std::endl;
        sparseCode->deviceVector = NULL;
        delete sparseCode;
        
       // std::cout<<"    # Total Time: "<<std::chrono::duration_cast<std::chrono::seconds>(tot1 + tot2).count()<<" s"<<std::endl<<std::endl;
    }

    auto start = std::chrono::steady_clock::now();

    //std::cout<<"Final OMP"<<std::endl;
    sparseCode = omp->work(noisePatches, dictionary);
    delete omp;

    auto end = std::chrono::steady_clock::now();
    auto tot1 = end - start;
    //std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(tot1).count()<<" s"<<std::endl<<std::endl;        
}

//************************************
//  Buils Image from denoised patches
//***********************************
void CudaKSvdDenoiser::createImage(bool transpose){

    //Build deNoised Patches/Images
    delete noisePatches;

    //std::cout<<"Build image denoised"<<std::endl;
    auto start = std::chrono::steady_clock::now();

    CuBlasMatrixMult* mult= (CuBlasMatrixMult*) MatrixOps::factory(CUBLAS_MULT);
    mult->setOps(CUBLAS_OP_N, CUBLAS_OP_N);
    noisePatches = mult->work(dictionary, sparseCode);   
    
    noisePatches->copyOnHost(); 

    host_vector<float>* img = new host_vector<float>(inputMatrix->m * inputMatrix->n,0);
    host_vector<float> blockWeight(dictionary->m,1);
    host_vector<float> imgWeight(inputMatrix->m * inputMatrix->n,0);

    int patchIdx = 0 ;
    for(int i = 0; i + patchWidthDim <= inputMatrix->m; i = i + slidingWidth){
        
        for(int j = 0; j + patchHeightDim <= inputMatrix->n; j = j + slidingHeight){

            int startPatchIdx = (i*inputMatrix->n) + j;
            int colIdx = 0;

            for(int k = startPatchIdx; k < startPatchIdx + patchWidthDim * inputMatrix->n; k += inputMatrix->n){

                transform(noisePatches->hostVector->begin() + (patchIdx * dictionary->m) + (colIdx * patchHeightDim),
                          noisePatches->hostVector->begin() + (patchIdx * dictionary->m) + ((colIdx + 1) * patchHeightDim),
                          img->begin() + k,
                          img->begin() + k,
                          plus<float>());
                
                transform(blockWeight.begin() + colIdx * patchHeightDim,
                          blockWeight.begin() + (colIdx + 1) * patchHeightDim,
                          imgWeight.begin() + k,
                          imgWeight.begin() + k,
                          plus<float>());
                colIdx++ ;
            }
            patchIdx++;
        }
    }

    host_vector<float> temp(img->size());
    transform(img->begin(),
              img->end(),
              imgWeight.begin(),
              temp.begin(),
              _1 / _2);
    
    transform(temp.begin(), temp. end(),inputMatrix->hostVector->begin(),temp.begin(), minus<float>());
    float d = transform_reduce(temp.begin(), temp.end(), square<float>(), 0.f, plus<float>());
    d /= sigma * temp.size();

    d = abs(sqrtf(d) -1.f);

    transform(inputMatrix->hostVector->begin(),
              inputMatrix->hostVector->end(),
              img->begin(),
              img->begin(),
              (_1 * d) + _2);

    transform(img->begin(),
              img->end(),
              imgWeight.begin(),
              img->begin(),
              _1 / (d +_2));
    
    if(speckle)
        transform(img->begin(), img->end(), img->begin(), myExp<float>());

    CImg<float>* image = new CImg<float>(inputMatrix->m, inputMatrix->n);   
    image->_data = img->data();
    
    if(transpose)
        image->transpose();

    outputMatrix = new Matrix(inputMatrix->m, inputMatrix->n, inputMatrix->m, image->_data);

    auto end = std::chrono::steady_clock::now();
    auto tot1 = end - start;

    //std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(tot1).count()<<" s"<<std::endl<<std::endl;
}

//**************************************************************
//  Buils Image from each subImage (patches with 0 overlapping)
//*************************************************************
void CudaKSvdDenoiser::createImageFromSubImages(Matrix* patches, Matrix* input){

    host_vector<float>* img = new host_vector<float>(input->m * input->n,0);

    int patchesXcolumn = input->n / subImageWidthDim;

    for(int patchIdx = 0 ; patchIdx < patches->n; patchIdx += patchesXcolumn){

        int stride = patchIdx * patches->m;

        for(int i = 0; i < patchesXcolumn; i++){

            int stride2 = patchesXcolumn * subImageWidthDim;
            for(int j = 0; j < subImageHeightDim; j++){

                int startPatches = patches->m * (i + patchIdx) + (j * subImageWidthDim);
                copy(patches->hostVector->begin() + startPatches, patches->hostVector->begin() + startPatches + subImageWidthDim, img->begin() + stride + (stride2 * j) + (i * subImageWidthDim));
            }
        }
    }

   
    CImg<float>* image = new CImg<float>(input->m, input->n);   
    image->_data = img->data();
    image->transpose();
    outputMatrix = new Matrix(input->m, input->n, input->m, image->_data);
}