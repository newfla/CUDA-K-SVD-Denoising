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
        ~Matrix();
        Matrix* clone();
        static Matrix* randomMatrix(int, int, int);
    
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
        std::vector<Matrix*> getOutputMatrices();
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
        std::vector<Matrix*> output;
        
        virtual void init(Matrix*) ;
        virtual void work() = 0;
        virtual std::vector<Matrix*> getOutputMatrices() = 0;
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
        virtual std::vector<Matrix*> getOutputMatrices();

};

class svd::CuSolverGeSvd : public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvd();
        void init(Matrix*);
        void work();
        std::vector<Matrix*> getOutputMatrices();

    private:
        float* deviceRWork = NULL;

    friend SvdEngine;
};

class svd::CuSolverGeSvdJ: public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvdJ();
        void init(Matrix*);
        void work();
        std::vector<Matrix*> getOutputMatrices();

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