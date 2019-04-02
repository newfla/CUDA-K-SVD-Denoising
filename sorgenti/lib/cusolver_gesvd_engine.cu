#include <svdLib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace svd;

void CuSolverGeSvd::init(Matrix* matrix){

    //Call parent method
    SvdCudaEngine::init(matrix);

    //Allocate WorkSpace on device
    cusolverDnSgesvd_bufferSize(cusolverH, matrix->m, matrix->n, &lWork);
    cudaMalloc((void**) &deviceWork , sizeof(double)*lWork);
    
}

void CuSolverGeSvd::work(){

    //DGESVD
    cusolverDnSgesvd(
        cusolverH,
        'A',
        'A',
        input->m,
        input->n,
        deviceA,
        input->ld,
        deviceS,
        deviceU,
        input->m,
        deviceVT,
        input->n,
        deviceWork,
        lWork,
        deviceRWork,
        deviceInfo
    );
    cudaDeviceSynchronize();
}

std::vector<Matrix*> CuSolverGeSvd::getOutputMatrices(){
    cudaFree(deviceInfo);
    cudaFree(deviceRWork);
    return SvdCudaEngine::getOutputMatrices(); 
}