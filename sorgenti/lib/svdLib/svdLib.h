#if !defined(SVD_H)

#include <chrono>
#include <iostream>
#include <cusolverDn.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <utilityLib.h>

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
        void setMatrix(utl::Matrix*);
        thrust::host_vector<utl::Matrix*> getOutputMatrices();
        thrust::device_vector<utl::Matrix*> getDeviceOutputMatrices();
        utl::TimeElapsed* getTimeElapsed();

    private:
        utl::TimeElapsed* timeElapsed = NULL;
        SvdEngine* svdEngine = NULL;
};

class svd::SvdEngine{

    public:
        virtual ~SvdEngine();
        static SvdEngine* factory(SvdEngineType type);

    protected:
        utl::Matrix *input = NULL;
        thrust::host_vector<utl::Matrix*> output;
        
        virtual void init(utl::Matrix*) ;
        virtual void work() = 0;
        virtual thrust::host_vector<utl::Matrix*> getOutputMatrices() = 0;
        virtual thrust::device_vector<utl::Matrix*> getDeviceOutputMatrices() = 0;
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
        virtual void init(utl::Matrix*);
        virtual thrust::host_vector<utl::Matrix*> getOutputMatrices();
        thrust::device_vector<utl::Matrix*> getDeviceOutputMatrices();

};

class svd::CuSolverGeSvd : public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvd();
        void init(utl::Matrix*);
        void work();
        thrust::host_vector<utl::Matrix*> getOutputMatrices();
        thrust::device_vector<utl::Matrix*> getDeviceOutputMatrices();

    private:
        float* deviceRWork = NULL;

    friend SvdEngine;
};

class svd::CuSolverGeSvdJ: public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvdJ();
        void init(utl::Matrix*);
        void work();
        thrust::host_vector<utl::Matrix*> getOutputMatrices();
        thrust::device_vector<utl::Matrix*> getDeviceOutputMatrices();

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