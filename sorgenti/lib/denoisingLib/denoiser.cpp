#include <denoisingLib.h>

using namespace denoising;
using namespace svd;
using namespace baseUtl;
using namespace thrust;
using namespace cimg_library;

Denoiser::Denoiser(){

    timeElapsed = new TimeElapsed();
    psnr = new host_vector<double>(2);
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

    if(psnr!=NULL)
        delete psnr;
}

//**************************
//  Load image
//  output:  + staus (bool)
//*************************
bool Denoiser::loadImage(){

    auto start = std::chrono::steady_clock::now();

    inputImage = new CImg<float>(inputFile.c_str());

    sigma = (float) sqrt(inputImage->variance_noise());

    inputImage->transpose();

    if(inputImage==NULL)
        return false;

    inputMatrix = new Matrix(inputImage->height(), inputImage->width(), inputImage->height(), inputImage->data());//inputImage->RGBtoYCbCr().channel(0).data());
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
  
    CImg<float> image(outputMatrix->n, outputMatrix->ld);   

    image._data = outputMatrix->hostVector->data();
    image.normalize(0.,255.);

    try{
        image.save(outputFile.c_str());
    }catch(CImgArgumentException e){return false;}

    auto end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    if(refImage.back() == '/'){
        psnr->data()[0] = -1;
        psnr->data()[1] = -1;
        return true;
    }
    
    CImg<float> ref(refImage.c_str()), old(inputFile.c_str());
    psnr->data()[0] = ref.PSNR(old);
    psnr->data()[1] = ref.PSNR(image);

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
        case CUDA_K_GESVDA:
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

//**********************************************************
//  Obtain time stats
//  output:  + timer (TimeElapsed*) ms timers foreach phase
//*********************************************************
baseUtl::TimeElapsed* Denoiser::getTimeElapsed(){
    return timeElapsed;
}

//***************************************************************************
//  Obtain PSNR stats
//  output:  + psnr (host_vector<double>*) PSNR before/after denoising image
//**************************************************************************
host_vector<double>* Denoiser::getPsnr(){
    return psnr;
}