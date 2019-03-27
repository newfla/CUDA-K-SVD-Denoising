#include <svdLib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace svd;

void CuSolverDnDgeSvd::init(Matrix* matrix){

    //Call parent method
    SvdCudaEngine::init(matrix);

    //Allocate Space on device
    cusolverDnDgesvd_bufferSize(cusolverH, matrix->m, matrix->n, &lWork);
    cudaMalloc ((void**)&deviceInfo, sizeof(int));
    cudaMalloc((void**) &deviceWork , sizeof(double)*lWork);


}

void CuSolverDnDgeSvd::work(){

    //DGESVD
    cusolverDnDgesvd(
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
    cudaMemcpy(&infoGpu, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost);

}

std::vector<Matrix*> CuSolverDnDgeSvd::getOutputMatrices(){
    cudaFree(deviceInfo);
    cudaFree(deviceRWork);
    return SvdCudaEngine::getOutputMatrices(); 
}