#if !defined(DENOISING_H)
#include <string>
#include <algorithm>
#include <dirent.h>
#include <CImg.h>
#include <svdLib.h>

namespace denoising{

    //Class section
    class Denoiser;
    class CudaSvdDenoiser;
    class BatchDenoiser;

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
        virtual bool loadImage();
        virtual bool saveImage();
        virtual bool internalDenoising() = 0;
        Denoiser();

    private:
        std::string inputFile, outputFile;
        svd::Matrix *inputMatrix = NULL, *outputMatrix = NULL;
        svd::TimeElapsed* timeElapsed = NULL;
        cimg_library::CImg<float> *inputImage = NULL;

    friend BatchDenoiser;
};

class denoising::CudaSvdDenoiser : public denoising::Denoiser{

    public:
        ~CudaSvdDenoiser();
        signed char denoising();

    protected:
        bool loadImage();
        bool saveImage();
        bool internalDenoising();

    private:
        DenoiserType type;

        CudaSvdDenoiser();

    friend Denoiser;
};

class denoising::BatchDenoiser{

    public:
        ~BatchDenoiser();
        std::vector<svd::TimeElapsed*> getTimeElapsed();
        std::vector<signed char> seqBatchDenoising();
        static BatchDenoiser* factory(DenoiserType, std::string, std::string);

    protected:
        std::vector<svd::TimeElapsed*> times;
        std::vector<Denoiser*> denoisers;

    private:
        BatchDenoiser();
};

#endif