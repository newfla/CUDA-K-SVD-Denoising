#if !defined(SVD_H)
#include <cstdint>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <cusolverDn.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace svd{

    //Class section
    class Matrix;
    class TimeElapsed;
    class SvdEngine;
    class SvdContainer;
    class SvdCudaEngine;
    class CuSolverGeSvd;
    class CuSolverGeSvdJ;

    //Enum section
    enum SvdEngineType{CUSOLVER_GESVD, CUSOLVER_GESVDJ};
};

class svd::Matrix{

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

class svd::TimeElapsed{

    public:
        int64_t init, working, finalize;

        int64_t getTotalTime();
};

class svd::SvdContainer{

    public:
        SvdContainer(SvdEngine*);
        ~SvdContainer();
        void setMatrix(Matrix*);
        thrust::host_vector<Matrix*> getOutputMatrices();
        TimeElapsed* getTimeElapsed();

    private:
        TimeElapsed* timeElapsed = NULL;
        SvdEngine* svdEngine = NULL;
};

class svd::SvdEngine{

    public:
        virtual ~SvdEngine();
        static SvdEngine* factory(SvdEngineType type);

    protected:
        Matrix *input = NULL;
        thrust::host_vector<Matrix*> output;
        
        virtual void init(Matrix*) ;
        virtual void work() = 0;
        virtual thrust::host_vector<Matrix*> getOutputMatrices() = 0;
        SvdEngine();

        friend SvdContainer;
};

class svd::SvdCudaEngine : public svd::SvdEngine{
    protected:
        float *deviceA , *deviceU, *deviceS, *deviceVT, *deviceWork;
        int lWork = 0;
        int *deviceInfo;
        cusolverDnHandle_t cusolverH;

        SvdCudaEngine();
        virtual void init(Matrix*);
        virtual thrust::host_vector<Matrix*> getOutputMatrices();

};

class svd::CuSolverGeSvd : public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvd();
        void init(Matrix*);
        void work();
        thrust::host_vector<Matrix*> getOutputMatrices();

    private:
        float* deviceRWork = NULL;

    friend SvdEngine;
};

class svd::CuSolverGeSvdJ: public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvdJ();
        void init(Matrix*);
        void work();
        thrust::host_vector<Matrix*> getOutputMatrices();

    private:
        float tolerance;
        int maxSweeps;
        int econ = 0;
        gesvdjInfo_t gesvdjParams;
        cusolverEigMode_t jobZ = CUSOLVER_EIG_MODE_VECTOR;

        void printStat();

    friend SvdEngine;

};

#endif