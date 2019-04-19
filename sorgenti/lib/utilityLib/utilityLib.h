#if !defined(UTILITY_H)

#include <chrono>
#include <cstdint>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

namespace utl{

    //Class section
    class Matrix;
    class TimeElapsed;
    class MatrixOps;
    class CuBlasMatrixMult;
    class CuBlasMatrixAdd;

    //Enum section
    enum MatrixOpsType{CUBLAS_MULT, CUBLAS_ADD};
};

class utl::Matrix{

    public:
        int m, n, ld;
        thrust::host_vector<float> *hostVector = NULL;
        thrust::device_vector<float> *deviceVector = NULL;

        Matrix(int, int, int, float*);
        Matrix(int, int, int, thrust::host_vector<float>*);
        Matrix(int, int, int, thrust::device_vector<float>*);
        ~Matrix();
        Matrix* cloneHost();
        void copyOnDevice();
        void copyOnHost();
        static Matrix* randomMatrixHost(int, int, int); 

    private:
        Matrix();

};

class utl::TimeElapsed{

    public:
        int64_t init, working, finalize;

        int64_t getTotalTime();
};

class utl::MatrixOps{

    public:
        virtual utl::Matrix* work(Matrix* a, Matrix* b) = 0;
        virtual ~MatrixOps();
        utl::TimeElapsed* getTimeElapsed();
        void setCoeff(float, float);
        static MatrixOps* factory(MatrixOpsType);

    protected:
        utl::Matrix* a;
        utl::Matrix* b;
        utl::Matrix* c;
        utl::TimeElapsed* timeElapsed = new utl::TimeElapsed();
        float alfa = 1;
        float beta = 0;
        
        MatrixOps();
        virtual void init() = 0;
        virtual void finalize() = 0;

};

class utl::CuBlasMatrixMult : public utl::MatrixOps{

    public:
        utl::Matrix* work(Matrix* a, Matrix* b);
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

class utl::CuBlasMatrixAdd : public utl::MatrixOps{

    public:
        utl::Matrix* work(Matrix* a, Matrix* b);
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

#endif