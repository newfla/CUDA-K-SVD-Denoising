#if !defined(UTILITY_H)

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

namespace utl{

    //Class section
    class Matrix;
    class TimeElapsed;
    class MatrixMult;
    class CuBlassMatrixMult;

    //Enum section
    enum MatrixMultType{CUBLASS_MULT};
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

class utl::MatrixMult{

    public:
        virtual utl::Matrix* multiply() = 0;
        virtual ~MatrixMult();
        utl::TimeElapsed* getTimeElapsed();
        static MatrixMult* factory(MatrixMultType, utl::Matrix*, utl::Matrix* ,int , int);

    protected:
        utl::Matrix* a;
        utl::Matrix* b;
        utl::Matrix* c;
        utl::TimeElapsed* timeElapsed = new utl::TimeElapsed();
        float alfa = 1;
        float beta = 0;
        
        MatrixMult();
        virtual void init() = 0;
        virtual void finalize() = 0;

};

class utl::CuBlassMatrixMult : public utl::MatrixMult{

    public:
        utl::Matrix* multiply();
        
    protected:
        CuBlassMatrixMult();
        void init();
        void finalize();

    private:
        cublasHandle_t handle;
        cublasOperation_t op = CUBLAS_OP_N;
        thrust::device_vector<float>* cVector; 
    
    friend MatrixMult;

};

#endif