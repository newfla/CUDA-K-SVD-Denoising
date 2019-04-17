#include <denoisingLib.h>

using namespace denoising;
using namespace svd;
using namespace utl;
using namespace cimg_library;

Denoiser::Denoiser(){

    timeElapsed = new TimeElapsed();
}
//**************
//  Destructor
//  Free images 
//*************
Denoiser::~Denoiser(){
    
    if(inputImage!=NULL)
       delete inputImage;
    
    //outputImage is alreay freed in saveImage

    if(timeElapsed!=NULL)
        delete timeElapsed;
}

//**************************
//  Load image
//  output:  + staus (bool)
//*************************
bool Denoiser::loadImage(){

    auto start = std::chrono::steady_clock::now();

    inputImage = new CImg<float>(inputFile.c_str());

    if(inputImage==NULL)
        return false;

    inputMatrix = new Matrix(inputImage->height(), inputImage->width(), inputImage->height(), inputImage->RGBtoYCbCr().channel(0).data());

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return true;
}

//**************************
//  Save image
//  output:  + staus (bool)
//*************************
bool Denoiser::saveImage(){

    auto start = std::chrono::steady_clock::now();

    //TODO da rimuovere
    outputMatrix = inputMatrix;

    CImg<float> image(outputMatrix->n, outputMatrix->ld);

    //TODO in realtà sarebbe deviceVector
    image._data = outputMatrix->hostVector->data();

    //TODO da rimuovere
    image._is_shared = true;

    image.normalize(0,255);

    try{
        image.save(outputFile.c_str());
    }catch(CImgArgumentException e){return false;}

    auto end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return true;
}

//************************************************************************
//  Denoiser Factory method instantiates an object based on type
//  input:  + type (DenoiserType) of denoisers that will be used
//          + inputFile path from which load image
//          + outputFile path where save image   
//  output: + Denoiser (Denoiser*)
//***********************************************************************
Denoiser* Denoiser::factory(DenoiserType type, std::string inputFile, std::string outputFile){
    Denoiser* denoiser = NULL;
    switch (type)
    {
        case CUDA_K_GESVD:
        case CUDA_K_GESVDJ:
            denoiser = new CudaKSvdDenoiser();
            ((CudaKSvdDenoiser*) denoiser)-> type = type;
            break;
            
        default:
            return NULL;
    }

    if(denoiser != NULL){
        denoiser->inputFile = inputFile;
        denoiser->outputFile = outputFile;
    }

    return denoiser;
}

utl::TimeElapsed* Denoiser::getTimeElapsed(){
    return timeElapsed;
}