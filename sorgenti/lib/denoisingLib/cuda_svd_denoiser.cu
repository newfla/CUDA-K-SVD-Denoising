#include <denoisingLib.h>

using namespace denoising;
using namespace svd;
using namespace baseUtl;
using namespace matUtl;
using namespace thrust;
using namespace thrust::placeholders;


CudaKSvdDenoiser::CudaKSvdDenoiser(){
}

CudaKSvdDenoiser::~CudaKSvdDenoiser(){
    //TODO
}

//********************
//  Istantiate SvdObj
//*******************
void CudaKSvdDenoiser::buildSvdContainer(){
    switch (type)
        {
            case CUSOLVER_GESVD:
                svdContainer = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVD));
                break;

            default:
            case CUSOLVER_GESVDJ:
                svdContainer = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVDJ));
                break;
         
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

    //TODO
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
    host_vector<float>* patches = new host_vector<float>();

    //Create patch division on host

    for(i = 0; i + patchSquareDim <= inputMatrix->n; i+= slidingPatch){ //n = ImageWidth

        for(j = 0; j + patchSquareDim <= inputMatrix->m; j+= slidingPatch){ // m = ImageHeight

            host_vector<float> patch;
            int startPatch = i * inputMatrix->m + j;

            for(int k = startPatch; k < startPatch + patchSquareDim * inputMatrix->m; k += inputMatrix->m)
                patch.insert(patch.end(), inputMatrix->hostVector->begin() + k, inputMatrix->hostVector->begin() + k + patchSquareDim);

            patches->insert(patches->end(), patch.begin(), patch.end()); 
        }  
    }

    i = patchSquareDim * patchSquareDim;
    j = patches->size() / i;
    noisePatches = new Matrix(i, j, i, patches);
    
    std::cout<<"    # Patches: "<<j<<"  Dim: "<<i<<std::endl;

    //Copy data on device
    noisePatches->copyOnDevice();

    auto end = std::chrono::steady_clock::now();
    std::cout<<"    # Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.<<" s"<<std::endl<<std::endl;
}

//*************************************************************************************************************
//  Init a dictionary using #atoms square patches column major of fixed dims (patchSquareDim x patcSquareDim)
//************************************************************************************************************
void CudaKSvdDenoiser::initDictionary(){

    std::cout<<"Init Dictionary"<<std::endl;

    auto start = std::chrono::steady_clock::now();
    int dim = patchSquareDim * patchSquareDim;
    device_vector<float> * dict = new device_vector<float>();

    //Copy Patches and normalization using norm2
    for (int i = 0; i < atoms; i++){
        
        //Copy a single patch
        dict->insert(dict->end(), noisePatches->deviceVector->begin() + (i * dim), noisePatches->deviceVector->begin() + ((i + 1) * dim));

        //Calculate norm
        float norm = sqrtf(transform_reduce(dict->begin() + (i * dim), dict->begin() + ((i+1) * dim), mySquare<float>(), 0, myPlus<float>()));

        //Normalize vector
        transform(dict->begin() + (i * dim), dict->begin() + ((i + 1) * dim), dict->begin() + (i * dim), _1/norm);
    }

    dictionary = new Matrix(dim, atoms, dim, dict);
    //std::cout<<dict->size();

    auto end = std::chrono::steady_clock::now();
    std::cout<<"    # Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<" ms"<<std::endl<<std::endl;
}

//******************************************************
//  Update dictionary columns using SVD on Error Matrix
//*****************************************************
void CudaKSvdDenoiser::updateDictionary(){

    int dim = patchSquareDim * patchSquareDim;
    minus<float> binaryOp;
 
    for(int atomIdx = 0 ; atomIdx < sparseCode->m ; atomIdx++){ //->m = # atoms
        std::cout<<"Atom idx:   "<<atomIdx<<std::endl;

        device_vector<int> relevantDataIndices;
        MatrixOps* mult;
        Matrix* dx;
        Matrix* v;
        Matrix* u;
        Matrix* s;
        float bestS;
        buildSvdContainer();

        //Find for each patch relevant atoms --> idx!=0 
		for(int i = 0; i < sparseCode->n; i++){ //-> n = #NoisePatches

			if(sparseCode->deviceVector->data()[(i * sparseCode->m) + atomIdx] != 0) 
				relevantDataIndices.push_back(i); 
		}

        std::cout<<"relevantIndices.size: "<<relevantDataIndices.size()<<std::endl;

        //Only update atom shared by 1 or more pacthes
        if(relevantDataIndices.size()<1)
            continue;

        //Collect input (patches and coeffs) that used this atom
        device_vector<float> selectInput;//relevantDataIndices.size() * dim);
		device_vector<float> selectSparseCode;//relevantDataIndices.size() * sparseCode->m);

        for(int inputIdx : relevantDataIndices) {
			selectInput.insert(selectInput.end(),noisePatches->deviceVector->begin() + inputIdx * dim, noisePatches->deviceVector->begin() + (inputIdx+1) * dim); 
			selectSparseCode.insert(selectSparseCode.end(),sparseCode->deviceVector->begin() + inputIdx * sparseCode->m, sparseCode->deviceVector->begin() + (inputIdx + 1) * sparseCode->m); 
		}

        //Remove atom from dict --> coef at row atomIdx = 0

        for(int i = 0; i < relevantDataIndices.size(); i++)
            selectSparseCode[(i * sparseCode->m) + atomIdx] = 0;

        //DX = Dictionary * selectSparseCode
        mult = MatrixOps::factory(CUBLAS_MULT);
        ((CuBlasMatrixMult*)mult)->setOps(CUBLAS_OP_N, CUBLAS_OP_N);
        dx = mult->work(dictionary, new Matrix(sparseCode->m , relevantDataIndices.size(), sparseCode->m, &selectSparseCode));

        std::cout<<"Dopo di moltiplicare"<<std::endl;

        //E = coff - dx
        device_vector<float> error(selectInput.size());

        transform(selectInput.begin(), selectInput.end(), dx->deviceVector->begin(), error.begin(), binaryOp);

        std::cout<<"Prima transform andata"<<std::endl;

        //Compute SVD on E
        svdContainer->setMatrix(new Matrix(dim, relevantDataIndices.size(), dim, &error));
        host_vector<Matrix*> usvt = svdContainer->getDeviceOutputMatrices();

        std::cout<<"SvD ANdato"<<std::endl;
        
        //Traspose V
      /*  tras = MatrixOps::factory(CUBLAS_ADD);
        ((CuBlasMatrixAdd*)tras)->setOps(CUBLAS_OP_T, CUBLAS_OP_T);
        v = tras->work(usvt[2], usvt[2]);*/

        //Replace dictionary column
        u = usvt[0];        
        transform(u->deviceVector->begin(), u->deviceVector->begin() + u->m, u->deviceVector->begin(), _1 * -1.f);
        std::cout<<"Seconda transform andata"<<std::endl;
        copy(u->deviceVector->begin(), u->deviceVector->begin() + u->m, dictionary->deviceVector->begin() + atomIdx * dim);

        //Calculate new coeffs
        s = usvt[1];
        v = usvt[2];
        bestS = s->deviceVector->data()[0];
        transform(v->deviceVector->begin(), v->deviceVector->begin() + v->m, v->deviceVector->begin(), _1 * -1.f * bestS);
        std::cout<<"Terza transform andata"<<std::endl;

        for(int i = 0 ; i < relevantDataIndices.size() ; i++ ) {
            int inputIdx = relevantDataIndices[i];
            sparseCode->deviceVector->data()[inputIdx * sparseCode->m + atomIdx] = v->deviceVector->data()[i] ; 
         }

        std::cout<<"FIn iter"<<std::endl;        
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
       // std::cout<<"SparseCode size: "<< sparseCode->deviceVector->size()<<std::endl;
        auto end = std::chrono::steady_clock::now();
        auto tot1 = end - start;
        std::cout<<"    # OMP Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot1).count()/1000.<<" s"<<std::endl;

        //Dict update phase
        start = std::chrono::steady_clock::now();
        updateDictionary();
        end = std::chrono::steady_clock::now();
        auto tot2 = end - start;
        std::cout<<"    # Dict update Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot2).count()/1000.<<" s"<<std::endl;

        delete sparseCode;
        
        std::cout<<"    # Total time: "<<tot1.count() + tot2.count()<<std::endl<<std::endl; 
    }

    //Compute last Iter sparseCode
    delete sparseCode;

    auto start = std::chrono::steady_clock::now();

    CuBlasMatrixOmp* omp = (CuBlasMatrixOmp*) MatrixOps::factory(CUBLAS_OMP);
    sparseCode = omp->work(noisePatches, dictionary);

    auto end = std::chrono::steady_clock::now();
    auto tot1 = end - start;
    std::cout<<"Last iter OMP Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot1).count()<<" ms"<<std::endl;        
}

void CudaKSvdDenoiser::createImage(){

    //Build deNoised Patches
    delete noisePatches;
    
    auto start = std::chrono::steady_clock::now();

    std::cout<<"Matrix mult between Dict and sparseCode";

    CuBlasMatrixMult* mult= (CuBlasMatrixMult*) MatrixOps::factory(CUBLAS_MULT);
    mult->setOps(CUBLAS_OP_N, CUBLAS_OP_N);
    noisePatches = mult->work(dictionary, sparseCode);

    auto end = std::chrono::steady_clock::now();
    auto tot1 = end - start;
    std::cout<<"    Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot1).count()<<" ms"<<std::endl;


    //Build DenoisedImage
    start = std::chrono::steady_clock::now();

    std::cout<<"Build image denoised";

    device_vector<float>* img = new device_vector<float>(inputMatrix->m * inputMatrix->n);
    device_vector<float> blockWeight(patchSquareDim*patchSquareDim,1);
    device_vector<float> imgWeight(inputMatrix->m * inputMatrix->n);

    int dim = patchSquareDim * patchSquareDim;
    int patchIdx = 0 ;
    for(int i = 0; i + patchSquareDim <= inputMatrix->n; i = i + slidingPatch){
        
        for(int j = 0; j + patchSquareDim <= inputMatrix->m; j = j + slidingPatch){

            int startPatchIdx = i*inputMatrix->m + j ;
            int colIdx = 0;

            device_vector<float> thisPatch(noisePatches->deviceVector->begin() + patchIdx*dim, noisePatches->deviceVector->begin() + (patchIdx + 1)*dim); 

            for(int k = startPatchIdx; k < startPatchIdx + patchSquareDim*inputMatrix->m; k += inputMatrix->m){

                std::transform(thisPatch.begin() + colIdx*patchSquareDim, thisPatch.begin() + (colIdx +1)*patchSquareDim,
                           img->begin() + k, img->begin() + k, std::plus<float>());
                
                std::transform(blockWeight.begin() + colIdx*patchSquareDim ,blockWeight.begin() + (colIdx + 1)*patchSquareDim,
                           imgWeight.begin() + k , imgWeight.begin() + k, std::plus<float>());
            colIdx++ ;
            }
        }
        patchIdx++;
    }

    for(int i = 0 ; i < img->size(); i++)
		img->data()[i] = (inputMatrix->deviceVector->data()[i] + 0.034 * 0.25 * img->data()[i])/(1 + 0.034 * 0.25 * imgWeight[i]); 

    outputMatrix = new Matrix(inputMatrix->m, inputMatrix->n, inputMatrix->m, img);

    end = std::chrono::steady_clock::now();
    tot1 = end - start;

    std::cout<<"    Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot1).count()<<" ms"<<std::endl;

}