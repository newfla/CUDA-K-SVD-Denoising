#if !defined(SVD_H)
#include <cstdint>
namespace svd{

    //Class section
    class Matrix;
    class TimeElapsed;
    class SvdEngine;
    class SvdContainer;


    //Enum section
    enum SvdEngineType{ CUSOLVER_DN_DGESVD};
};

class svd::Matrix{

    public:
        int m, n, ld;
        double* matrix;

        ~Matrix();
        static Matrix* randomMatrix(int,int,int);
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
        Matrix* getOutputMatrices();
        TimeElapsed* getTimeElapsed();

    private:
        TimeElapsed* timeElapsed;
        SvdEngine* svdEngine;
};

class svd::SvdEngine{

    public:
        static SvdEngine factory(SvdEngineType);

    private:
        Matrix *input, *output;
        
        virtual void init(Matrix*);
        virtual void work();
        virtual Matrix* getOutputMatrices();

        friend SvdContainer;
};

#endif