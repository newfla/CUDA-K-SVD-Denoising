#if !defined(MAT_UTILITY_H)

#include <chrono>
#include <cstdint>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <cublas_v2.h>
#include <svdLib.h>
namespace matUtl{

    //Class section
    class MatrixOps;
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
        virtual void finalize() = 0;

};

class matUtl::CuBlasMatrixMult : public matUtl::MatrixOps{

    public:
        baseUtl::Matrix* work(baseUtl::Matrix* a, baseUtl::Matrix* b);
        void setOps(cublasOperation_t, cublasOperation_t);
        
    protected:
        CuBlasMatrixMult();
        void init();
        void finalize();

    private:
        cublasHandle_t handle;
        cublasOperation_t op1, op2;
        thrust::device_vector<float>* cVector; 
    
    friend MatrixOps;

};

class matUtl::CuBlasMatrixAdd : public matUtl::MatrixOps{

    public:
        baseUtl::Matrix* work(baseUtl::Matrix* a, baseUtl::Matrix* b);
        void setOps(cublasOperation_t, cublasOperation_t);
        
    protected:
        CuBlasMatrixAdd();
        void init();
        void finalize();

    private:
        cublasHandle_t handle;
        cublasOperation_t op1, op2;
        thrust::device_vector<float>* cVector; 
    
    friend MatrixOps;

};

class matUtl::CuBlasMatrixOmp : public matUtl::MatrixOps{

    public:
        baseUtl::Matrix* work(baseUtl::Matrix* a, baseUtl::Matrix* b);
        baseUtl::Matrix* work2(baseUtl::Matrix* a, baseUtl::Matrix* b);
        void setLimits(float, int);

    protected:
        CuBlasMatrixOmp();
        void init();
        void finalize();

    private:
        cublasHandle_t handle;
        thrust::device_vector<float>* sparseCode;
        int maxIters = 5;

    friend MatrixOps;
};

#endif