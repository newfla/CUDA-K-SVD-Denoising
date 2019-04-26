#if !defined(SVD_H)

#include <chrono>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <baseUtilityLib.h>

namespace svd{

    //Class section
    class SvdEngine;
    class SvdContainer;
    class SvdCudaEngine;
    class CuSolverGeSvd;
    class CuSolverGeSvdJ;

    //Enum section
    enum SvdEngineType{CUSOLVER_GESVD, CUSOLVER_GESVDJ};
};

class svd::SvdContainer{

    public:
        SvdContainer(SvdEngine*);
        ~SvdContainer();
        void setMatrix(baseUtl::Matrix*);
        thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        thrust::device_vector<baseUtl::Matrix*> getDeviceOutputMatrices();
        baseUtl::TimeElapsed* getTimeElapsed();

    private:
        baseUtl::TimeElapsed* timeElapsed = NULL;
        SvdEngine* svdEngine = NULL;
};

class svd::SvdEngine{

    public:
        virtual ~SvdEngine();
        static SvdEngine* factory(SvdEngineType type);

    protected:
        baseUtl::Matrix *input = NULL;
        thrust::host_vector<baseUtl::Matrix*> output;
        
        virtual void init(baseUtl::Matrix*) ;
        virtual void work() = 0;
        virtual thrust::host_vector<baseUtl::Matrix*> getOutputMatrices() = 0;
        virtual thrust::device_vector<baseUtl::Matrix*> getDeviceOutputMatrices() = 0;
        SvdEngine();

        friend SvdContainer;
};

class svd::SvdCudaEngine : public svd::SvdEngine{
    protected:
        float *deviceA , *deviceU, *deviceS, *deviceVT, *deviceWork;
        int lWork = 0, less = 0;
        int *deviceInfo;
        cusolverDnHandle_t cusolverH;

        SvdCudaEngine();
        virtual void init(baseUtl::Matrix*);
        virtual thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        thrust::device_vector<baseUtl::Matrix*> getDeviceOutputMatrices();

};

class svd::CuSolverGeSvd : public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvd();
        void init(baseUtl::Matrix*);
        void work();
        thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        thrust::device_vector<baseUtl::Matrix*> getDeviceOutputMatrices();

    private:
        float* deviceRWork = NULL;

    friend SvdEngine;
};

class svd::CuSolverGeSvdJ: public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvdJ();
        void init(baseUtl::Matrix*);
        void work();
        thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        thrust::device_vector<baseUtl::Matrix*> getDeviceOutputMatrices();

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