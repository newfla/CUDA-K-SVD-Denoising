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
void CudaKSvdDenoiser::buildSvdContainer(){
    switch (type)
        {
            case CUDA_K_GESVD:
                svdContainer = new SvdContainer(SvdEngine::factory(CUSOLVER_GESVD));
                break;

            default:
            case CUDA_K_GESVDJ:
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

            device_vector<float> patch;
            int startPatch = (i * inputMatrix->m) + j;

            for(int k = startPatch; k < startPatch + patchSquareDim * inputMatrix->m; k += inputMatrix->m)
                patch.insert(patch.end(), inputMatrix->deviceVector->begin() + k, inputMatrix->deviceVector->begin() + k + patchSquareDim);

            patches->insert(patches->end(), patch.begin(), patch.end()); 
        }  
    }

    i = patchSquareDim * patchSquareDim;
    j = patches->size() / i;
    noisePatches = new Matrix(i, j, i, patches);
    
    std::cout<<"    # Patches: "<<j<<"  Dim: "<<i<<std::endl;

   // delete inputMatrix->deviceVector;
    //inputMatrix->deviceVector = NULL;
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
        Matrix* u;
        Matrix* s;

        buildSvdContainer();

        //Find for each patch relevant atoms --> idx!=0 
		for(int i = 0; i < sparseCode->n; i++){ //-> n = #NoisePatches

			if(sparseCode->deviceVector->data()[(i * sparseCode->m) + atomIdx] != 0) 
				relevantDataIndices.push_back(i); 
               // std::cout<<sparseCode->deviceVector->data()[(i * sparseCode->m) + atomIdx]<<std::endl;
		}
		//std::cout<<"atomidx: "<<atomIdx<<std::endl;
		//std::cout<<"relevantIndices.size: "<<relevantDataIndices.size()<<std::endl;

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

        for(int i = 0; i < relevantDataIndices.size(); i++)
            selectSparseCode[(i * sparseCode->m) + atomIdx] = 0;

        //DX = Dictionary * selectSparseCode
        mult = MatrixOps::factory(CUBLAS_MULT);
        ((CuBlasMatrixMult*)mult)->setOps(CUBLAS_OP_N, CUBLAS_OP_N);
        dx = mult->work(dictionary, new Matrix(sparseCode->m , relevantDataIndices.size(), sparseCode->m, &selectSparseCode));        

        //dx = coff - dx
        transform(selectInput.begin(), selectInput.end(), dx->deviceVector->begin(),dx->deviceVector->begin(), minus<float>());

        //Compute SVD on E
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
        u = usv[0];        
        transform(u->deviceVector->begin(),
                  u->deviceVector->begin() + u->m,
                  dictionary->deviceVector->begin() + (atomIdx * dictionary->m),
                  _1 * -1.f);

        //Calculate new coeffs
        s = usv[1];
        /*for(int i = 0 ; i < v->m; i++)
            std::cout<<v->deviceVector->data()[i]<<std::endl;*/
        transform(v->deviceVector->begin(), v->deviceVector->begin() + v->m, v->deviceVector->begin(), -1.f * s->deviceVector->data()[0] * _1);

        for(int i = 0 ; i < relevantDataIndices.size() ; i++ ) {
            int inputIdx = relevantDataIndices[i];
            sparseCode->deviceVector->data()[(inputIdx * sparseCode->m) + atomIdx] = v->deviceVector->data()[i] ; 
            //std::cout<<"Sparse["<<i<<"]= "<<sparseCode->deviceVector->data()[(inputIdx * sparseCode->m) + atomIdx]<<std::endl;
        }
        /*for(int i =0; i< dictionary->m; i++)
            std::cout<<"Dict["<<i<<"] = "<<dictionary->deviceVector->data()[(atomIdx * dictionary->m)+i]<<std::endl;*/
      
        delete u;
        delete s;
        delete v;        
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

    //Build deNoised Patches
    delete noisePatches;
    
    auto start = std::chrono::steady_clock::now();

    std::cout<<"Matrix mult between Dict and sparseCode"<<std::endl;

    /*for(int i = 0; i < dictionary->deviceVector->size(); i++)
        std::cout<<"dict i: "<<i<<" val: "<<dictionary->deviceVector->data()[i]<<std::endl;

    std::cin.get();

    for(int i = 0; i < sparseCode->deviceVector->size(); i++)
        std::cout<<"sparse i: "<<i<<" val: "<<sparseCode->deviceVector->data()[i]<<std::endl;
    
    std::cin.get();*/

    CuBlasMatrixMult* mult= (CuBlasMatrixMult*) MatrixOps::factory(CUBLAS_MULT);
    mult->setOps(CUBLAS_OP_N, CUBLAS_OP_N);
    noisePatches = mult->work(dictionary, sparseCode);

    /*for(int i = 0; i < noisePatches->deviceVector->size(); i++)
        std::cout<<"img i: "<<i<<" val: "<<noisePatches->deviceVector->data()[i]<<std::endl;

    std::cin.get();*/


    auto end = std::chrono::steady_clock::now();
    auto tot1 = end - start;
    std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot1).count()<<" ms"<<std::endl<<std::endl;


    //Build DenoisedImage
    start = std::chrono::steady_clock::now();

    std::cout<<"Build image denoised"<<std::endl;

    device_vector<float>* img = new device_vector<float>(inputMatrix->m * inputMatrix->n,0);
    device_vector<float> blockWeight(dictionary->m,1);
    device_vector<float> imgWeight(inputMatrix->m * inputMatrix->n,0);

    int patchIdx = 0 ;
    for(int i = 0; i + patchSquareDim <= inputMatrix->n; i = i + slidingPatch){
        
        for(int j = 0; j + patchSquareDim <= inputMatrix->m; j = j + slidingPatch){

            int startPatchIdx = (i*inputMatrix->m) + j;
            int colIdx = 0;

           // device_vector<float> thisPatch(noisePatches->deviceVector->begin() + (patchIdx*dictionary->m), noisePatches->deviceVector->begin() + ((patchIdx + 1)*dictionary->m)); 

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
    /*for(int i = 0; i < imgWeight.size(); i++)
      std::cout<<"imgWeight i:"<<i<<" val: "<<imgWeight[i]<<std::endl;
std::cin.get();
   for(int i = 0; i < img->size(); i++)
      std::cout<<"img i:"<<i<<" val: "<<img->data()[i]<<std::endl;*/

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

   // for(int i = 0; i < img->size(); i++)
    //  std::cout<<"img i:"<<i<<" val: "<<img->data()[i]<<std::endl;

    outputMatrix = new Matrix(inputMatrix->m, inputMatrix->n, inputMatrix->m, img);

    CuBlasMatrixAdd* tras = (CuBlasMatrixAdd*) MatrixOps::factory(CUBLAS_ADD);
    tras->setOps(CUBLAS_OP_T, CUBLAS_OP_T);
    std::swap(outputMatrix->m, outputMatrix->n);
    outputMatrix->ld = outputMatrix->m;
    Matrix* trs = tras->work(outputMatrix, outputMatrix);

    delete outputMatrix;
    outputMatrix = trs;
    outputMatrix->copyOnHost();
 //   for(int i=0; i<outputMatrix->deviceVector->size(); i++)
  //    std::cout<<"i: "<<i<<" val: "<<outputMatrix->deviceVector->data()[i]<<std::endl;
    end = std::chrono::steady_clock::now();
    tot1 = end - start;

    std::cout<<"    # Time Elapsed: "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot1).count()<<" ms"<<std::endl;

}