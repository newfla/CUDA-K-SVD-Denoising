#if !defined(SVD_H)
#include <cstdint>
#include <random>
#include <vector>
#include <iostream> //TODO da togliere poi
#include <cusolverDn.h>

namespace svd{

    //Class section
    class Matrix;
    class TimeElapsed;
    class SvdEngine;
    class SvdContainer;
    class SvdCudaEngine;
    class CuSolverDnDgeSvd;


    //Enum section
    enum SvdEngineType{CUSOLVER_DN_DGESVD};
};

class svd::Matrix{

    public:
        int m, n, ld;
        double* matrix;

        Matrix(int, int, int, double*);
        ~Matrix();
        static Matrix* randomMatrix(int, int, int);
    
    private:
        Matrix();
};

class svd::TimeElapsed{

    public:
        int64_t getInitTime();
        int64_t getWorkingTime();
        int64_t getFinalizeTime();
        int64_t getTotalTime();

    private:
        int64_t init, working, finalize;

    friend SvdContainer;
};

class svd::SvdContainer{

    public:
        SvdContainer(SvdEngine*);
        ~SvdContainer();
        void setMatrix(Matrix*);
        std::vector<Matrix*> getOutputMatrices();
        TimeElapsed* getTimeElapsed();

    private:
        TimeElapsed* timeElapsed;
        SvdEngine* svdEngine;
};

class svd::SvdEngine{

    public:
        virtual ~SvdEngine();
        static SvdEngine* factory(SvdEngineType type);

    protected:
        Matrix *input;
        std::vector<Matrix*> output;
        cusolverDnHandle_t cusolverH;
        
        virtual void init(Matrix*) ;
        virtual void work() = 0;
        virtual std::vector<Matrix*> getOutputMatrices() = 0;

        friend SvdContainer;
};

class svd::SvdCudaEngine : public svd::SvdEngine{
    protected:
        double *deviceA, *deviceU, *deviceS, *deviceVT, *deviceWork;
        int lWork = 0;
        //cusolverDnHandle_t cusolverH;

        virtual void init(Matrix*);
        virtual std::vector<Matrix*> getOutputMatrices();

};

class svd::CuSolverDnDgeSvd : public svd::SvdCudaEngine{

    public:
        void init(Matrix*);
        void work();
        std::vector<Matrix*> getOutputMatrices();

    private:
        double* deviceRWork;
        int *deviceInfo, infoGpu=0;

};

#endif