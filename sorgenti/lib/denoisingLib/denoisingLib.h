#if !defined(DENOISING_H)

#include <jsonxx.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <CImg.h>
#include <matUtilityLib.h>

namespace denoising{

    //Class section
    class Denoiser;
    class CudaKSvdDenoiser;
    class BatchDenoiser;

    //Enum section
    enum DenoiserType{CUDA_K_GESVD, CUDA_K_GESVDJ, CUDA_K_GESVDA};
};

class denoising::Denoiser{

    public:
        virtual ~Denoiser();
        static Denoiser* factory(DenoiserType, std::string, std::string);
        virtual signed char denoising() = 0;
        baseUtl::TimeElapsed* getTimeElapsed();
        thrust::host_vector<double>* getPsnr();
        thrust::host_vector<double>* getEnl();

    protected:
        int patchWidthDim = 8;
        int patchHeightDim = 8;
        int slidingWidth = 2;
        int slidingHeight = 2;
        int subImageWidthDim = 0;
        int subImageHeightDim = 0;
        int ompIter = 5;
        int atoms = 256;
        int iter = 10;
        int minOmpIterBatch = 0;
        float sigma;
        bool speckle = false;
        std::string refImage;
        baseUtl::Matrix* inputMatrix = NULL;
        baseUtl::Matrix* outputMatrix = NULL;
        baseUtl::TimeElapsed* timeElapsed = NULL;
        thrust::host_vector<double> * psnr = NULL;
        thrust::host_vector<double> * enl = NULL;
        
        Denoiser();
        virtual bool loadImage();
        virtual bool saveImage();
        virtual bool internalDenoising() = 0;

    private:
        std::string inputFile, outputFile;
        cimg_library::CImg<float> *inputImage = NULL;

    friend BatchDenoiser;

};

class denoising::CudaKSvdDenoiser : public denoising::Denoiser{

    public:
        ~CudaKSvdDenoiser();
        signed char denoising();

    protected:
        bool loadImage();
        bool saveImage();
        bool internalDenoising();

    private:
        int blocks;
        DenoiserType type;
        baseUtl::Matrix* noisePatches = NULL;
        baseUtl::Matrix* dictionary = NULL;
        baseUtl::Matrix* sparseCode = NULL;
        
        CudaKSvdDenoiser();
        svd::SvdContainer* buildSvdContainer();
        void createPatches(bool);
        void initDictionary();
        void updateDictionary();
        void createImage(bool);
        void createImageFromSubImages(baseUtl::Matrix*, baseUtl::Matrix*);
        void kSvd();
        void checkEnl(bool,  thrust::host_vector<float>*);

    friend Denoiser;
    friend BatchDenoiser;
};

class denoising::BatchDenoiser{

    public:
        ~BatchDenoiser();
        thrust::host_vector<baseUtl::TimeElapsed*> getTimeElapsed();
        thrust::host_vector<thrust::host_vector<double>*> getPsnr();
        thrust::host_vector<thrust::host_vector<double>*> getEnl();  
        thrust::host_vector<signed char> seqBatchDenoising();
        static BatchDenoiser* factory(DenoiserType, std::string);

    protected:
        thrust::host_vector<baseUtl::TimeElapsed*> times;
        thrust::host_vector<thrust::host_vector<double>*> psnrs;
        thrust::host_vector<thrust::host_vector<double>*> enls;
        thrust::host_vector<Denoiser*> denoisers;

    private:
        BatchDenoiser();
};

template <typename T>
struct myLog
{
  __host__ __device__
    T operator()(const T& x) const{
        if(x == 0)
            return 0.;
        return log2f(x);
    }
};

template <typename T>
struct myExp
{
  __host__ __device__
    T operator()(const T& x) const{
        return powf(2.,x);
    }
};

#endif