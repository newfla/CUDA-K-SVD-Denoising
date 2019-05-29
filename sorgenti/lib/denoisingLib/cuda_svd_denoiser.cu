#include <denoisingLib.h>

using namespace denoising;
using namespace baseUtl;
using namespace svd;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;
using namespace cimg_library;


CudaKSvdDenoiser::CudaKSvdDenoiser(){}

CudaKSvdDenoiser::~CudaKSvdDenoiser(){}

//**************************
//  Istantiate SvdCOntainer
//*************************
SvdContainer* CudaKSvdDenoiser::buildSvdContainer(){
    
    switch (type)
        {
            case CUDA_K_GESVD:
                return new SvdContainer(SvdEngine::factory(CUSOLVER_GESVD));

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
   // if(a)
        //transform(inputMatrix->hostVector->begin(),inputMatrix->hostVector->end(),inputMatrix->hostVector->begin(),myLog());
    return a;
}

//**************************
//  Save image
//  output:  + staus (bool)
//*************************
bool CudaKSvdDenoiser::saveImage(){
    return Denoiser::saveImage();
}

//**************************
//  CUDA K-SVD implementation 
//  output:  + staus (bool)
//*************************
bool CudaKSvdDenoiser::internalDenoising(){

    auto start = std::chrono::steady_clock::now();

    //Divide image in patches column major of fixed dims
    createPatches();

    //Init Dict
    initDictionary();

    //Start #iter K-SVD
    kSvd();

    //Rebuild originalImage
    createImage();

    CuBlasMatrixOps::finalize();
    SvdCudaEngine::finalize();
    cudaDeviceReset();
    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return true;
}

//**********************************************************************************************
//  Divide image in square patches column major of fixed dims (patchWidthDim x patchHeightDim)
//*********************************************************************************************
void CudaKSvdDenoiser::createPatches(){

    std::cout<<"Create Patches"<<std::endl;

    auto start = std::chrono::steady_clock::now();

    int i, j;
    host_vector<float>* patchesHost = new host_vector<float>();

    //Create patch division on host

    for(int i = 0; i + patchHeightDim <= inputMatrix->n; i+= slidingHeight){ //n = ImageWidth

        for(j = 0; j + patchWidthDim <= inputMatrix->m; j+= slidingWidth){ // m = ImageHeight

            int startPatch = (i * inputMatrix->m) + j;

            for(int k = startPatch; k < startPatch + patchHeightDim * inputMatrix->m; k += inputMatrix->m)
               patchesHost->insert(patchesHost->end(), inputMatrix->hostVector->begin() + k, inputMatrix->hostVector->begin() + k + patchWidthDim);
        }  
    }

    i = patchWidthDim * patchHeightDim;
    j = patchesHost->size() / i;
    noisePatches = new Matrix(i, j, i, patchesHost);

   /* for (int i = 0; i < noisePatches->n; i++)
    {
        for (int j = 0; j < noisePatches->m; j++)
            std::cout<<noisePatches->hostVector->data()[i *noisePatches->m + j]<<std::endl;
        
        std::cin.get();
        
    }*/
    

    noisePatches->copyOnDevice();
    //noisePatches->hostVector = NULL;
    
    std::cout<<"    # Patches: "<<j<<"  Dim: "<<i<<std::endl;

    std::swap(inputMatrix->m, inputMatrix->n);
    inputMatrix->ld = inputMatrix->m;
    
    auto end = std::chrono::steady_clock::now();
    std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(end-start).count()<<" s"<<std::endl<<std::endl;
}

//*************************************************************************************************************
//  Init a dictionary using #atoms patches column major of fixed dims (patchWidthDim x patchHeightDim)
//************************************************************************************************************
void CudaKSvdDenoiser::initDictionary(){

    std::cout<<"Init Dictionary"<<std::endl;

    auto start = std::chrono::steady_clock::now();
    int dim = patchWidthDim * patchHeightDim;
    host_vector<float> * dict = new host_vector<float>(noisePatches->hostVector->begin(), noisePatches->hostVector->begin() + dim * atoms);

    //Normalize patches using norm2
    for (int i = 0; i < atoms; i++){

        //Calculate norm
        float norm = sqrtf(transform_reduce(dict->begin() + (i * dim), dict->begin() + ((i+1) * dim), square<float>(), 0, plus<float>()));

        //Normalize vector
        transform(dict->begin() + (i * dim), dict->begin() + ((i + 1) * dim), dict->begin() + (i * dim), _1/norm);
    }

    dictionary = new Matrix(dim, atoms, dim, dict);
    dictionary->copyOnDevice();

    auto end = std::chrono::steady_clock::now();
    std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<" ms"<<std::endl<<std::endl;
}

//******************************************************
//  Update dictionary columns using SVD on Error Matrix
//*****************************************************
void CudaKSvdDenoiser::updateDictionary(){

    for(int atomIdx = 0 ; atomIdx < sparseCode->m ; atomIdx++){ //->m = # atoms
        
        device_vector<int> relevantDataIndices;
        MatrixOps* mult;
        Matrix* dx;
        Matrix* v;

        buildSvdContainer();

        //Find patches that used current atom --> idx!=0 
        for(int i = 0; i < sparseCode->n; i++){ //-> n = #NoisePatches
			if(sparseCode->deviceVector->data()[(i * sparseCode->m) + atomIdx] != 0) 
				relevantDataIndices.push_back(i); 
		}

        //Only update atom shared by 1 or more patches
        if(relevantDataIndices.size()<1)
            continue;

        //Collect input (patches and coeffs) that used this atom
        device_vector<float> selectInput;
		device_vector<float> selectSparseCode;

        for(int inputIdx : relevantDataIndices) {
			selectInput.insert(selectInput.end(),noisePatches->deviceVector->begin() + (inputIdx * dictionary->m), noisePatches->deviceVector->begin() + ((inputIdx+1) * dictionary->m)); 
			selectSparseCode.insert(selectSparseCode.end(),sparseCode->deviceVector->begin() + (inputIdx * sparseCode->m), sparseCode->deviceVector->begin() + ((inputIdx + 1) * sparseCode->m)); 
		}

        //Remove atom from dict --> coef at row atomIdx = 0
        device_vector<int> idxSeq(relevantDataIndices.size());
        sequence(idxSeq.begin(), idxSeq.end(), atomIdx, sparseCode->m);
        transform(make_permutation_iterator(selectSparseCode.begin(), idxSeq.begin()),
                  make_permutation_iterator(selectSparseCode.end(), idxSeq.end()),
                  make_permutation_iterator(selectSparseCode.begin(), idxSeq.begin()),
                  _1 = 0.);

        //DX = Dictionary * selectSparseCode
        mult = MatrixOps::factory(CUBLAS_MULT);
        ((CuBlasMatrixMult*)mult)->setOps(CUBLAS_OP_N, CUBLAS_OP_N);
        dx = mult->work(dictionary, new Matrix(sparseCode->m , relevantDataIndices.size(), sparseCode->m, &selectSparseCode));        

        //dx = coff - dx
        transform(selectInput.begin(), selectInput.end(), dx->deviceVector->begin(),dx->deviceVector->begin(), minus<float>());

        //Compute SVD on E
        SvdContainer* svdContainer = buildSvdContainer();
        svdContainer->setMatrix(dx);
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
        transform(relevantDataIndices.begin(),relevantDataIndices.end(), idxSeq.begin(),  atomIdx + (_1 * sparseCode->m));        
       
        transform(v->deviceVector->begin(),
                  v->deviceVector->begin() + relevantDataIndices.size(),
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

    for(int i = 0 ; i < iter ; i++){

        std::cout<<"Ksvd-Iter: "<<i+1<<std::endl;

        //OMP phase
        auto start = std::chrono::steady_clock::now();
     
        CuBlasMatrixOmp* omp = (CuBlasMatrixOmp*) MatrixOps::factory(CUBLAS_OMP);
        omp->maxIters = ompIter;
        sparseCode = omp->work2(noisePatches, dictionary);

        auto end = std::chrono::steady_clock::now();
        auto tot1 = end - start;
        std::cout<<"    # OMP Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(tot1).count()<<" s"<<std::endl;

        //Dict update phase
        start = std::chrono::steady_clock::now();
        updateDictionary();
        end = std::chrono::steady_clock::now();
        auto tot2 = end - start;
        std::cout<<"    # Dict update Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(tot2).count()<<" s"<<std::endl;
        delete sparseCode;
        
        std::cout<<"    # Total Time: "<<std::chrono::duration_cast<std::chrono::seconds>(tot1 + tot2).count()<<" s"<<std::endl<<std::endl;
    }

    auto start = std::chrono::steady_clock::now();

    std::cout<<"Final OMP"<<std::endl;
    CuBlasMatrixOmp* omp = (CuBlasMatrixOmp*) MatrixOps::factory(CUBLAS_OMP);
    sparseCode = omp->work2(noisePatches, dictionary);

    auto end = std::chrono::steady_clock::now();
    auto tot1 = end - start;
    std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(tot1).count()<<" s"<<std::endl<<std::endl;        
}

void CudaKSvdDenoiser::createImage(){

    //Build deNoised Patches/Images
    delete noisePatches;

    std::cout<<"Build image denoised"<<std::endl;
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
    d /= sigma * sigma * temp.size();

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

   /* transform(inputMatrix->hostVector->begin(),
              inputMatrix->hostVector->end(),
              img->begin(),
              img->begin(),
              (_1 + 0.034 * sigma *_2));
    
    transform(img->begin(),
              img->end(),
              imgWeight.begin(),
              img->begin(),
              _1 / (1. + 0.034 * sigma *_2));*/

    
   // transform(img->begin(),img->end(),img->begin(),myPow());

    CImg<float>* image = new CImg<float>(inputMatrix->m, inputMatrix->n);   
    image->_data = img->data();
    image->transpose();

    outputMatrix = new Matrix(inputMatrix->m, inputMatrix->n, inputMatrix->m, image->_data);
 
    auto end = std::chrono::steady_clock::now();
    auto tot1 = end - start;

    std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot1).count()<<" ms"<<std::endl<<std::endl;
}