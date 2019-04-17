#include <denoisingLib.h>
#include<iostream> //TODO da togliere
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

using namespace denoising;
using namespace svd;
using namespace utl;
using namespace thrust;
using namespace thrust::placeholders;


CudaKSvdDenoiser::CudaKSvdDenoiser(){}

CudaKSvdDenoiser::~CudaKSvdDenoiser(){
    //TODO
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

    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return true;
}

//**********************************************************************************************
//  Divide image in square patches column major of fixed dims (patchSquareDime x patcSquareDim)
//*********************************************************************************************
void CudaKSvdDenoiser::createPatches(){

    std::cout<<"Creating patches";

    auto start = std::chrono::steady_clock::now();

    int i, j;
    host_vector<float>* patches = new host_vector<float>();

    //Create patch division on host

    for(i = 0; i + patchSquareDim < inputMatrix->n; i+= slidingPatch){ //n = ImageWidth

        for(j = 0; j + patchSquareDim < inputMatrix->m; j+= slidingPatch){ // m = ImagaeHeight

            host_vector<float> patch;
            int startPatch = i * inputMatrix->m + j;

            for(int k = startPatch; k < startPatch + patchSquareDim * inputMatrix->m; k++)
                patch.insert(patch.end(), inputMatrix->hostVector->begin() + k, inputMatrix->hostVector->begin() + k + patchSquareDim);

            patches->insert(patches->end(), patch.begin(), patch.end()); 
        }  
    }

    i = patchSquareDim * patchSquareDim;
    j = patches->size() / i;

    noisePatches = new Matrix(i, j, i, patches);

    //Copy data on device
    noisePatches->copyOnDevice();

    auto end = std::chrono::steady_clock::now();
    std::cout<<"    Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<" ms"<<std::endl;
}

//*************************************************************************************************************
//  Init a dictionary using #atoms square patches column major of fixed dims (patchSquareDime x patcSquareDim)
//************************************************************************************************************
void CudaKSvdDenoiser::initDictionary(){

    std::cout<<"Init Dictionary"<<std::endl;

    auto start = std::chrono::steady_clock::now();
    int dim = patchSquareDim * patchSquareDim;
    square<float> unaryOp;
    plus<float> binaryOp;
    device_vector<device_vector<float>* > container (atoms);
    device_vector<float> * dict = new device_vector<float>(atoms*dim);

    //copy patches
    for(int i=0; i<atoms; i++)
        container[i] = new device_vector<float> (noisePatches->deviceVector->begin() + i * dim, noisePatches->deviceVector->begin() + (i+1) * dim);

    //patch normalization using norm2
    for(device_vector<float>* patch: container){
        
        float norm = sqrtf(transform_reduce(patch->begin(), patch->end(), unaryOp, 0, binaryOp));

        transform(patch->begin(), patch->end(), patch->begin(), _1 /= norm);
    }       

    //Build first iter dictionary
    for(device_vector<float>* i : container){
        dict->insert(dict->end(), i->begin(), i->end());
        cudaFree(raw_pointer_cast(i->data()));
    }

    dictionary = new Matrix(dim, atoms, dim, dict);    

    
}

//*************************************************************************************************************
//  kSVD algorithm
//  GPU Version
//************************************************************************************************************
void CudaKSvdDenoiser::kSvd(){

    sparseCode = new Matrix(atoms, noisePatches->n, atoms, new device_vector<float>(atoms * noisePatches->n));

    for(int i = 0 ; i < iter ; i++){

        std::cout<<"Iter #: "<<iter<<std::endl;

        //OMP phase
        auto start = std::chrono::steady_clock::now();
        //omp
        auto end = std::chrono::steady_clock::now();
        auto tot1 = end -start;
        std::cout<<"    -OMP Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot1).count()<<" ms"<<std::endl;

        //Dict update phase
        start = std::chrono::steady_clock::now();
        updateDictionary();
        end = std::chrono::steady_clock::now();
        auto tot2 = end -start;
        std::cout<<"    -Dict update Time Elapsed : "<<std::chrono::duration_cast<std::chrono::milliseconds>(tot2).count()<<" ms"<<std::endl;
        
        std::cout<<"Total time: "<<tot1.count() + tot2.count()<<std::endl<<std::endl; 
    } 
}

void CudaKSvdDenoiser::updateDictionary(){
    int dim = patchSquareDim * patchSquareDim;
 
    for(int atomIdx = 0 ; atomIdx < sparseCode->m ; atomIdx++){

        device_vector<int> relevantDataIndices;
        device_vector<float> selectInput;
		device_vector<float> selectSparseCode;

        //Find atoms-patches associted used in sparseCode
		for(int i = 0 ; i < sparseCode->n ;i++){

			if(sparseCode->deviceVector->data()[i* sparseCode->m + atomIdx] != 0 ) 
				relevantDataIndices.push_back(i); 
		}

        //Only update atom shared by more than 1 inputData
        if(relevantDataIndices.size()<1)
            continue;

        //Collect patches and sparseData associted to this atom
        for(int inputIdx : relevantDataIndices) {
			selectInput.insert(selectInput.end(),noisePatches->deviceVector->begin() + inputIdx * dim, noisePatches->deviceVector->begin() + (inputIdx+1) * dim); 
			selectSparseCode.insert(selectSparseCode.end(),sparseCode->deviceVector->begin() + inputIdx * dim, sparseCode->deviceVector->begin() + (inputIdx + 1) * dim); 
		}

        //Remove atom from dict --> coef at row atomIdx = 0
        for(int i : relevantDataIndices)
            selectSparseCode[i * sparseCode->m + atomIdx] = 0;
        
            
    }


}