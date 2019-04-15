#if !defined(DENOISING_H)
#include <string>
#include <algorithm>
#include <dirent.h>
#include <CImg.h>
#include <svdLib.h>

namespace denoising{

    //Class section
    class Denoiser;
    class CudaKSvdDenoiser;
    class BatchDenoiser;
    class PowFunctor;

    //Enum section
    enum DenoiserType{CUDA_K_GESVD, CUDA_K_GESVDJ};
};

class denoising::Denoiser{

    public:
        virtual ~Denoiser();
        static Denoiser* factory(DenoiserType, std::string, std::string);
        virtual signed char denoising() = 0;
        svd::TimeElapsed* getTimeElapsed();

    protected:
        svd::Matrix *inputMatrix = NULL, *outputMatrix = NULL;

        virtual bool loadImage();
        virtual bool saveImage();
        virtual bool internalDenoising() = 0;
        Denoiser();

    private:
        std::string inputFile, outputFile;
        svd::TimeElapsed* timeElapsed = NULL;
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
        int patchSquareDim = 8;
        int slidingPatch = 2;
        int atoms = 256;
        DenoiserType type;
        svd::Matrix* noisePatches = NULL;
        svd::Matrix* dictionary = NULL;

        CudaKSvdDenoiser();
        void createPatches();
        void initDictionary();
        
    friend Denoiser;
};

class denoising::BatchDenoiser{

    public:
        ~BatchDenoiser();
        thrust::host_vector<svd::TimeElapsed*> getTimeElapsed();
        thrust::host_vector<signed char> seqBatchDenoising();
        static BatchDenoiser* factory(DenoiserType, std::string, std::string);

    protected:
        thrust::host_vector<svd::TimeElapsed*> times;
        thrust::host_vector<Denoiser*> denoisers;

    private:
        BatchDenoiser();
};

#endif