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

    //Enum section
    enum DenoiserType{CUDA_K_GESVD, CUDA_K_GESVDJ};
};

class denoising::Denoiser{

    public:
        virtual ~Denoiser();
        static Denoiser* factory(DenoiserType, std::string, std::string);
        virtual signed char denoising() = 0;
        utl::TimeElapsed* getTimeElapsed();

    protected:
        utl::Matrix* inputMatrix = NULL;
        utl::Matrix* outputMatrix = NULL;
        utl::TimeElapsed* timeElapsed = NULL;
        
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
        int patchSquareDim = 8;
        int slidingPatch = 2;
        int atoms = 256;
        int iter = 10;
        DenoiserType type;
        utl::Matrix* noisePatches = NULL;
        utl::Matrix* dictionary = NULL;
        utl::Matrix* sparseCode = NULL;
        svd::SvdContainer* svdContainer = NULL;
        
        CudaKSvdDenoiser();
        void createPatches();
        void initDictionary();
        void updateDictionary();
        void kSvd();

    friend Denoiser;
};

class denoising::BatchDenoiser{

    public:
        ~BatchDenoiser();
        thrust::host_vector<utl::TimeElapsed*> getTimeElapsed();
        thrust::host_vector<signed char> seqBatchDenoising();
        static BatchDenoiser* factory(DenoiserType, std::string, std::string);


    protected:
        thrust::host_vector<utl::TimeElapsed*> times;
        thrust::host_vector<Denoiser*> denoisers;

    private:
        BatchDenoiser();
};

#endif