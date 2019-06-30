#if !defined(MAT_UTILITY_H)

#include <chrono>
#include <cstdint>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <cublas_v2.h>
#include <svdLib.h>

namespace matUtl{

    //Class section
    class MatrixOps;
    class CuBlasMatrixOps;
    class CuBlasMatrixMult;
    class CuBlasMatrixAdd;
    class CuBlasMatrixOmp;
    
    //Enum section
    enum MatrixOpsType{CUBLAS_MULT, CUBLAS_ADD, CUBLAS_OMP};
};


class matUtl::MatrixOps{

    public:
        virtual baseUtl::Matrix* work(baseUtl::Matrix* a, baseUtl::Matrix* b) = 0;
        virtual ~MatrixOps();
        baseUtl::TimeElapsed* getTimeElapsed();
        void setCoeff(float, float);
        static MatrixOps* factory(MatrixOpsType);

    protected:
        baseUtl::Matrix* a;
        baseUtl::Matrix* b;
        baseUtl::Matrix* c;
        baseUtl::TimeElapsed* timeElapsed = new baseUtl::TimeElapsed();
        float alfa = 1;
        float beta = 0;
        
        MatrixOps();
        virtual void init() = 0;

};

class matUtl::CuBlasMatrixOps : public matUtl::MatrixOps{

    public:
        virtual baseUtl::Matrix* work(baseUtl::Matrix* a, baseUtl::Matrix* b) = 0;
        void setOps(cublasOperation_t, cublasOperation_t);
        static void finalize();

    protected:
        static cublasHandle_t* handle;
        cublasOperation_t op1, op2;
        thrust::device_vector<float>* cVector = NULL; 

        void init();
        CuBlasMatrixOps();

};

class matUtl::CuBlasMatrixMult : public matUtl::CuBlasMatrixOps{

    public:
        baseUtl::Matrix* work(baseUtl::Matrix* a, baseUtl::Matrix* b);

    protected:
        CuBlasMatrixMult();
        void init();
        void finalize();        

    friend MatrixOps;

};

class matUtl::CuBlasMatrixAdd : public matUtl::CuBlasMatrixOps{

    public:
        baseUtl::Matrix* work(baseUtl::Matrix* a, baseUtl::Matrix* b);
        
    protected:
        CuBlasMatrixAdd();
        void init();
        void finalize();
    
    friend MatrixOps;

};

class matUtl::CuBlasMatrixOmp : public matUtl::CuBlasMatrixOps{

    public:
        baseUtl::Matrix* work(baseUtl::Matrix* a, baseUtl::Matrix* b);
        int maxIters = 5;
        int minOmpIterBatch = 0;
        ~CuBlasMatrixOmp();

    protected:
        CuBlasMatrixOmp();
        void init();
        void finalize();
        
    private:
        static thrust::host_vector<cudaStream_t>* streams;
        const static int maxStreams = 16;
        int blocks = 0, subIter;
        size_t free_byte, total_byte;
        thrust::host_vector<int>* patchesIter = NULL;
        thrust::device_vector<float>* proj = NULL;
        thrust::device_vector<float>* projAbs = NULL;
        thrust::device_vector<float>* tempVec = NULL;
        thrust::device_vector<int>* maxs = NULL;
        thrust::device_vector<float>* alfaBeta = NULL;
        thrust::device_vector<int>* chosenAtomIdxList = NULL;
        thrust::device_vector<int>* chosenAtomIdxList2 = NULL;
        thrust::device_vector<float>* tempMatMult = NULL;
        thrust::device_vector<float>* pseudoInverse = NULL;
        thrust::device_vector<float>* weightList = NULL;
    friend MatrixOps;
};

struct abs_val
{
  __host__ __device__
  float operator()(float x)
  {
    return abs(x);
  }
};

#endif