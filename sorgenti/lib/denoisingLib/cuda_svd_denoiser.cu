#include <denoisingLib.h>
#include<iostream> //TODO da togliere
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

using namespace denoising;
using namespace svd;
using namespace thrust;
using namespace thrust::placeholders;


CudaKSvdDenoiser::CudaKSvdDenoiser(){}

CudaKSvdDenoiser::~CudaKSvdDenoiser(){
    //TODO
}

//***************************************************************************************************************************************************
//  Load denoising save
//  output:  + status (signed char) 0 = done, -1 = image loading failed, -2 = denoising failed, -3 = image saving failed
//**************************************************************************************************************************************************
signed char CudaKSvdDenoiser::denoising(){

    if(!loadImage())
        return -1;    
    
    //TODO

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

    //Divide image in square patches column major of fixed dims
    createPatches();

    //Init Dict
    initDictionary();

    //TODO
    return +1;
}

//**********************************************************************************************
//  Divide image in square patches column major of fixed dims (patchSquareDime x patcSquareDim)
//*********************************************************************************************
void CudaKSvdDenoiser::createPatches(){

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

    noisePatches = new Matrix(patches->size()/j, j, patches->size()/j, patches);

    //Copy data on device
    noisePatches->copyOnDevice();
}

//*************************************************************************************************************
//  Init a dictionary using #atoms square patches column major of fixed dims (patchSquareDime x patcSquareDim)
//************************************************************************************************************
void CudaKSvdDenoiser::initDictionary(){

    device_vector<device_vector<float>* > container (atoms);
    int dim = patchSquareDim * patchSquareDim;
    square<float> unaryOp;
    plus<float> binaryOp;

    //copy patches
    for(int i=0; i<atoms; i++)
        container[i] = new device_vector<float> (noisePatches->deviceVector->begin() + i * dim, noisePatches->deviceVector->begin() + (i+1) * dim);

    //patch normalization using norm2
    for(device_vector<float>* patch: container){
        
        float norm = sqrtf(transform_reduce(patch->begin(), patch->end(), unaryOp, 0, binaryOp));

        transform(patch->begin(), patch->end(), patch->begin(), _1 /= norm);
    }       

    //Assamble first iter dictionary
    device_vector<float> * dict = new device_vector<float>(atoms*dim);

    for(device_vector<float>* i : container){
        dict->insert(dict->end(), i->begin(), i->end());
        cudaFree(raw_pointer_cast(i->data()));
    }

    dictionary = new Matrix(atoms, dim, dim, dict);    
}