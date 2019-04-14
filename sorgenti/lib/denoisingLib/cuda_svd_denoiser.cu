#include <denoisingLib.h>
#include<iostream> //TODO da togliere

using namespace denoising;

CudaKSvdDenoiser::CudaKSvdDenoiser(){}

CudaKSvdDenoiser::~CudaKSvdDenoiser(){
    //TODO
}

signed char CudaKSvdDenoiser::denoising(){

    if(!loadImage())
        return -1;    
    
    //TODO

    if(!saveImage())
        return -3;

    return 0;
}

bool CudaKSvdDenoiser::loadImage(){

    //TODO
    return Denoiser::loadImage();
}

bool CudaKSvdDenoiser::saveImage(){

    //TODO
    return Denoiser::saveImage();
}

bool CudaKSvdDenoiser::internalDenoising(){

    createPatches();

    //TODO
    return +1;
}

void CudaKSvdDenoiser::createPatches(){

    std::vector<float> img(inputMatrix->deviceVector->data(), inputMatrix->deviceVector->data() + ((inputMatrix->m * inputMatrix->n)));

    for(int i = 0; i + patchSquareDim < inputMatrix->n; i+= slidingPatch){ //n = ImageWidth

        for(int j = 0; j + patchSquareDim < inputMatrix->m; j+= slidingPatch){ // m = ImagaeHeight

            std::vector<float> patch;
            int startPatch = i * inputMatrix->m + j;

            for(int k = startPatch; k < startPatch + patchSquareDim * inputMatrix->m; k++)

                patch.insert(patch.end(), img.begin() + k, img.begin() + k + patchSquareDim);

            
            
        }
        
    }
    
    
}