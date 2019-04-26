#if !defined(DENOISING_H)

#include <jsonxx.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <CImg.h>
#include <matUtilityLib.h>
#include <thrust/transform_reduce.h>

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
        baseUtl::TimeElapsed* getTimeElapsed();

    protected:
        int patchSquareDim = 8;
        int slidingPatch = 2;
        baseUtl::Matrix* inputMatrix = NULL;
        baseUtl::Matrix* outputMatrix = NULL;
        baseUtl::TimeElapsed* timeElapsed = NULL;
        
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
        int atoms = 256;
        int iter = 10;
        int sigma = 25;

        bool loadImage();
        bool saveImage();
        bool internalDenoising();

    private:
        DenoiserType type;
        baseUtl::Matrix* noisePatches = NULL;
        baseUtl::Matrix* dictionary = NULL;
        baseUtl::Matrix* sparseCode = NULL;
        svd::SvdContainer* svdContainer = NULL;
        
        CudaKSvdDenoiser();
        void createPatches();
        void initDictionary();
        void updateDictionary();
        void createImage();
        void kSvd();

    friend Denoiser;
    friend BatchDenoiser;
};

class denoising::BatchDenoiser{

    public:
        ~BatchDenoiser();
        thrust::host_vector<baseUtl::TimeElapsed*> getTimeElapsed();
        thrust::host_vector<signed char> seqBatchDenoising();
        static BatchDenoiser* factory(DenoiserType, std::string);


    protected:
        thrust::host_vector<baseUtl::TimeElapsed*> times;
        thrust::host_vector<Denoiser*> denoisers;

    private:
        BatchDenoiser();
};

template<typename T>
struct mySquare
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  typedef T argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>x*x</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__ T operator()(const T &x) const {return x*x;}
}; // end square

template<typename T>
struct myPlus
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  /*! Function call operator. The return value is <tt>lhs + rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs + rhs;}
}; // end plus


#endif