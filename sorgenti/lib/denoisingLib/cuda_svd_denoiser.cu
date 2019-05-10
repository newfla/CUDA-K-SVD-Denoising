#include <denoisingLib.h>

using namespace denoising;
using namespace baseUtl;
using namespace svd;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;


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

    //TODO
    return Denoiser::loadImage();
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

    //Divide image in square patches column major of fixed dims
    createPatches();

    //Init Dict
    initDictionary();

    //Start #iter K-SVD
    kSvd();

    //Rebuild originalImage
    createImage();

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return true;
}

//**********************************************************************************************
//  Divide image in square patches column major of fixed dims (patchSquareDime x patcSquareDim)
//*********************************************************************************************
void CudaKSvdDenoiser::createPatches(){

    std::cout<<"Create Patches"<<std::endl;

    auto start = std::chrono::steady_clock::now();

    int i, j;
    device_vector<float>* patches = new device_vector<float>();

    CuBlasMatrixAdd* tras = (CuBlasMatrixAdd*) MatrixOps::factory(CUBLAS_ADD);
    tras->setOps(CUBLAS_OP_T, CUBLAS_OP_T);
    std::swap(inputMatrix->m, inputMatrix->n);
    inputMatrix->ld = inputMatrix->m;
    Matrix* v = tras->work(inputMatrix, inputMatrix);

    delete inputMatrix;
    inputMatrix = v;

    //Create patch division on host

    for(int i = 0; i + patchSquareDim <= inputMatrix->n; i+= slidingPatch){ //n = ImageWidth

        for(j = 0; j + patchSquareDim <= inputMatrix->m; j+= slidingPatch){ // m = ImageHeight

            int startPatch = (i * inputMatrix->m) + j;

            for(int k = startPatch; k < startPatch + patchSquareDim * inputMatrix->m; k += inputMatrix->m)
               patches->insert(patches->end(), inputMatrix->deviceVector->begin() + k, inputMatrix->deviceVector->begin() + k + patchSquareDim);
        }  
    }

    i = patchSquareDim * patchSquareDim;
    j = patches->size() / i;
    noisePatches = new Matrix(i, j, i, patches);
    
    std::cout<<"    # Patches: "<<j<<"  Dim: "<<i<<std::endl;

    std::swap(inputMatrix->m, inputMatrix->n);
    inputMatrix->ld = inputMatrix->m;
    
    auto end = std::chrono::steady_clock::now();
    std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(end-start).count()<<" s"<<std::endl<<std::endl;
}

//*************************************************************************************************************
//  Init a dictionary using #atoms square patches column major of fixed dims (patchSquareDim x patcSquareDim)
//************************************************************************************************************
void CudaKSvdDenoiser::initDictionary(){

    std::cout<<"Init Dictionary"<<std::endl;

    auto start = std::chrono::steady_clock::now();
    int dim = patchSquareDim * patchSquareDim;
    device_vector<float> * dict = new device_vector<float>(noisePatches->deviceVector->begin(), noisePatches->deviceVector->begin() + dim * atoms);

    //Normalize patches using norm2
    for (int i = 0; i < atoms; i++){

        //Calculate norm
        float norm = sqrtf(transform_reduce(dict->begin() + (i * dim), dict->begin() + ((i+1) * dim), square<float>(), 0, plus<float>()));

        //Normalize vector
        transform(dict->begin() + (i * dim), dict->begin() + ((i + 1) * dim), dict->begin() + (i * dim), _1/norm);
    }

    dictionary = new Matrix(dim, atoms, dim, dict);

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

        //Find for each patch relevant atoms --> idx!=0 
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
        sparseCode = omp->work(noisePatches, dictionary);

        /*for (int i = 0; i < sparseCode->deviceVector->size(); i++){
            if(sparseCode->deviceVector->data()[i]!=0)    
                std::cout<<"sparse i: "<<i<<" val: "<<sparseCode->deviceVector->data()[i]<<std::endl;
        }
    	std::cin.get();*/

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
    sparseCode = omp->work(noisePatches, dictionary);

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

    device_vector<float>* img = new device_vector<float>(inputMatrix->m * inputMatrix->n,0);
    device_vector<float> blockWeight(dictionary->m,1);
    device_vector<float> imgWeight(inputMatrix->m * inputMatrix->n,0);

    int patchIdx = 0 ;
    for(int i = 0; i + patchSquareDim <= inputMatrix->n; i = i + slidingPatch){
        
        for(int j = 0; j + patchSquareDim <= inputMatrix->m; j = j + slidingPatch){

            int startPatchIdx = (i*inputMatrix->m) + j;
            int colIdx = 0;

            for(int k = startPatchIdx; k < startPatchIdx + patchSquareDim*inputMatrix->m; k += inputMatrix->m){

                transform(noisePatches->deviceVector->begin() + (patchIdx * dictionary->m) + (colIdx * patchSquareDim),
                          noisePatches->deviceVector->begin() + (patchIdx * dictionary->m) + (colIdx + 1) * patchSquareDim,
                          img->begin() + k,
                          img->begin() + k,
                          plus<float>());
                
                transform(blockWeight.begin() + colIdx * patchSquareDim,
                          blockWeight.begin() + (colIdx + 1) * patchSquareDim,
                          imgWeight.begin() + k,
                          imgWeight.begin() + k,
                          plus<float>());
                colIdx++ ;
            }
            patchIdx++;
        }
    }

    transform(inputMatrix->deviceVector->begin(),
              inputMatrix->deviceVector->end(),
              img->begin(),
              img->begin(),
              (_1 + 0.034 * sigma *_2));
    
    transform(img->begin(),
              img->end(),
              imgWeight.begin(),
              img->begin(),
              _1 / (1. + 0.034 * sigma *_2));

    outputMatrix = new Matrix(inputMatrix->m, inputMatrix->n, inputMatrix->m, img);

    CuBlasMatrixAdd* tras = (CuBlasMatrixAdd*) MatrixOps::factory(CUBLAS_ADD);
    tras->setOps(CUBLAS_OP_T, CUBLAS_OP_T);
    std::swap(outputMatrix->m, outputMatrix->n);
    outputMatrix->ld = outputMatrix->m;
    Matrix* trs = tras->work(outputMatrix, outputMatrix);

    delete outputMatrix;
    outputMatrix = trs;
    outputMatrix->copyOnHost();
 
    auto end = std::chrono::steady_clock::now();
    auto tot1 = end - start;

    std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::seconds>(tot1).count()<<" s"<<std::endl;
}