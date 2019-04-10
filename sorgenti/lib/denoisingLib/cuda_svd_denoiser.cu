#include <denoisingLib.h>
#include<iostream> //TODO da togliere

using namespace denoising;

CudaSvdDenoiser::CudaSvdDenoiser(){}

CudaSvdDenoiser::~CudaSvdDenoiser(){
    //TODO
}

signed char CudaSvdDenoiser::denoising(){

    if(!loadImage())
        return -1;    
    
    //TODO

    if(!saveImage())
        return -3;

    return 0;
}

bool CudaSvdDenoiser::loadImage(){

    //TODO
    return Denoiser::loadImage();
}

bool CudaSvdDenoiser::saveImage(){

    //TODO
    return Denoiser::saveImage();
}

bool CudaSvdDenoiser::internalDenoising(){

    //TODO
    return +1;
}