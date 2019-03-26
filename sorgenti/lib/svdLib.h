#if !defined(SVD_H)
#include <cstdint>
#include <random>
#include <vector>
#include <iostream> //TODO da togliere poi

namespace svd{

    //Class section
    class Matrix;
    class TimeElapsed;
    class SvdEngine;
    class SvdContainer;
    class CuSolverDnDgeSvd;


    //Enum section
    enum SvdEngineType{CUSOLVER_DN_DGESVD};
};

class svd::Matrix{

    public:
        int m, n, ld;
        double* matrix;

        ~Matrix();
        static Matrix* randomMatrix(int, int, int);
};

class svd::TimeElapsed{

    public:
    int x = 0;
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
        std::vector<Matrix*> output[3];
        
        virtual void init(Matrix*) ;
        virtual void work() = 0;
        virtual std::vector<Matrix*> getOutputMatrices() = 0;

        friend SvdContainer;
};

class svd::CuSolverDnDgeSvd : public svd::SvdEngine{

    public:
        void init(Matrix*);
        void work();
        std::vector<Matrix*> getOutputMatrices();

    private:
        double* deviceA;
};

#endif