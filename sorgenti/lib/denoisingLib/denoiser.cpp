#include <denoisingLib.h>

using namespace denoising;
using namespace svd;
using namespace cimg_library;

Denoiser::Denoiser(){
    timeElapsed = new TimeElapsed();
}

Denoiser::~Denoiser(){
    if(inputImage!=NULL) //Viene fatta la free anche del vettore in inputMatrix
       delete inputImage;

    //La free del vettore in outputMatrix viene fatta quando l'oggetto Cimage in saveImage viene distrutto

    if(timeElapsed!=NULL)
        delete timeElapsed;
}

bool Denoiser::loadImage(){
    auto start = std::chrono::steady_clock::now();

    inputImage = new CImg<float>(inputFile.c_str());

    if(inputImage==NULL)
        return false;

    inputMatrix = new svd::Matrix(inputImage->height(), inputImage->height(), inputImage->width(), inputImage->RGBtoYCbCr().channel(0).data());

    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return true;
}

bool Denoiser::saveImage(){

    auto start = std::chrono::steady_clock::now();

    //TODO da rimuovere
    outputMatrix = inputMatrix;

    CImg<float> image(outputMatrix->ld, outputMatrix->n);

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

svd::TimeElapsed* Denoiser::getTimeElapsed(){
    return timeElapsed;
}