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
    class CuSolverGeSvdJBatch;
    class CuSolverGeSvdABatch;

    //Enum section
    enum SvdEngineType{CUSOLVER_GESVD, CUSOLVER_GESVDJ, CUSOLVER_GESVDJ_BATCH, CUSOLVER_GESVDA_BATCH};
};

class svd::SvdContainer{

    public:
        SvdContainer(SvdEngine*);
        ~SvdContainer();
        void setMatrix(baseUtl::Matrix*);
        thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        thrust::host_vector<baseUtl::Matrix*> getDeviceOutputMatrices();
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
        virtual thrust::host_vector<baseUtl::Matrix*> getDeviceOutputMatrices() = 0;
        SvdEngine();

        friend SvdContainer;
};

class svd::SvdCudaEngine : public svd::SvdEngine{

    public:
        static void finalize();
        thrust::device_vector<float> *deviceA = NULL, *deviceU = NULL, *deviceS = NULL, *deviceVT = NULL;

    protected:
        float *deviceWork;
        int lWork = 0, less = 0;
        int *deviceInfo;
        static cusolverDnHandle_t* cusolverH;

        SvdCudaEngine();
        virtual void init(baseUtl::Matrix*);
        virtual thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        virtual thrust::host_vector<baseUtl::Matrix*> getDeviceOutputMatrices();
};

class svd::CuSolverGeSvd : public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvd();
        void init(baseUtl::Matrix*);
        void work();
        thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        thrust::host_vector<baseUtl::Matrix*> getDeviceOutputMatrices();

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
        thrust::host_vector<baseUtl::Matrix*> getDeviceOutputMatrices();

    private:
        float tolerance;
        int maxSweeps;
        int econ = 0;
        gesvdjInfo_t gesvdjParams;
        cusolverEigMode_t jobZ = CUSOLVER_EIG_MODE_VECTOR;

        void printStat();

    friend SvdEngine;

};

class svd::CuSolverGeSvdJBatch: public svd::SvdCudaEngine{

    protected:
        CuSolverGeSvdJBatch();
        void init(baseUtl::Matrix*);
        void work();
        thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        thrust::host_vector<baseUtl::Matrix*> getDeviceOutputMatrices();

    private:
        float tolerance;
        int maxSweeps;
        int econ = 0;
        gesvdjInfo_t gesvdjParams;
        cusolverEigMode_t jobZ = CUSOLVER_EIG_MODE_VECTOR;

        void printStat();

    friend SvdEngine;

};

class svd::CuSolverGeSvdABatch: public svd::SvdCudaEngine{

    public:
        thrust::host_vector<baseUtl::Matrix*> getDeviceOutputMatrices(int m, int n, int tot, thrust::device_ptr<float> ptr);

    protected:
        CuSolverGeSvdABatch();
        void init(baseUtl::Matrix*);
        void init(int m, int n, int tot, thrust::device_ptr<float> ptr);

        void work();
        thrust::host_vector<baseUtl::Matrix*> getOutputMatrices();
        thrust::host_vector<baseUtl::Matrix*> getDeviceOutputMatrices();

    private:
        cusolverEigMode_t jobZ = CUSOLVER_EIG_MODE_VECTOR;

    friend SvdEngine;

};

#endif