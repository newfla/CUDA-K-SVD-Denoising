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
    
    //outputImage is freed in saveImage

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

    if(inputImage==NULL)
        return false;

    inputImage->transpose();
    
    if(!bw)
         inputMatrix = new Matrix(inputImage->height(), inputImage->width(), inputImage->height(), inputImage->RGBtoYCbCr().channel(0).data());
    else
        inputMatrix = new Matrix(inputImage->height(), inputImage->width(), inputImage->height(), inputImage->data());

    if(speckle)
        transform(inputMatrix->hostVector->begin(),inputMatrix->hostVector->end(),inputMatrix->hostVector->begin(),myLog<float>());

    sigma = (float)inputImage->variance_noise();

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

    //LOW CONTRAST SAVED IMAGE? DON'T NORMALIZE AND USE OPENCV TO SAVE ON FILE
    //cv::Mat dest(outputMatrix->ld,outputMatrix->n,CV_32F,image._data);
    //cv::imwrite(outputFile, dest);

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
        denoiser->inputFile = inputFile;
        denoiser->outputFile = outputFile;

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

