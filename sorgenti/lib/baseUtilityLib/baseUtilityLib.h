#if !defined(BASE_UTILITY_H)

#include <chrono>
#include <cstdint>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace baseUtl{

    //Class section
    class Matrix;
    class TimeElapsed;
};

class baseUtl::Matrix{

    public:
        int m, n, ld;
        thrust::host_vector<float> *hostVector = NULL;
        thrust::device_vector<float> *deviceVector = NULL;

        Matrix(int, int, int, float*);
        Matrix(int, int, int, thrust::host_vector<float>*);
        Matrix(int, int, int, thrust::device_vector<float>*);
        Matrix(int, int, int, thrust::host_vector<float>*, thrust::device_vector<float>*);
        ~Matrix();
        Matrix* cloneHost();
        void copyOnDevice();
        void copyOnHost();
        static Matrix* randomMatrixHost(int, int, int); 

    private:
        Matrix();

};

class baseUtl::TimeElapsed{

    public:
        int64_t init, working, finalize;

        int64_t getTotalTime();
};


#endif