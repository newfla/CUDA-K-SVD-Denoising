#include <svdLib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace svd;

void SvdCudaEngine::init(Matrix* matrix){

    //Call parent method
    SvdEngine::init(matrix);

    //Create cusolverDn handle
    cusolverDnCreate(&cusolverH) ;

    //Alocate space for cusolverDnInfo
    cudaMalloc ((void**)&deviceInfo, sizeof(int));
    
    //Save matrix mem dimension
    size_t space = (matrix->ld)*(matrix->n)*sizeof(double);

    //Allocate memory on device
    //std::cout<<"risultato cudaMalloc di A: "<<
    cudaMalloc((void**) &deviceA, space);//<<std::endl;
    cudaMalloc((void**) &deviceU, (matrix->ld)*(matrix->m)*sizeof(double));
    cudaMalloc((void**) &deviceS, (matrix->n)*sizeof(double));
    cudaMalloc((void**) &deviceVT, (matrix->n)*(matrix->n)*sizeof(double));

    //Copy matrix on device
    //std::cout<<"risultato memcpy di A: "<<
    cudaMemcpy(deviceA, matrix->matrix, space, cudaMemcpyHostToDevice);//<<std::endl;

} 

std::vector<Matrix*> SvdCudaEngine::getOutputMatrices(){

    double *hostU, *hostVT, *hostS;
    Matrix *outputU, *outputVT, *outputS;

    //Cpu matrix resource allocation
    hostU = new double[(input->m)*(input->m)]();
    hostVT = new double[(input->n)*(input->n)]();
    hostS = new double[input->n]();

    //Output matrices
    outputU = new Matrix(input->ld, input->m, input->m, hostU);
    outputVT = new Matrix(input->n, input->n, input->n, hostVT);
    outputS = new Matrix (1, input->n, 1, hostS);

    //Copy back to host
    cudaMemcpy(hostU, deviceU, (outputU->ld)*(outputU->m)*sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(hostVT, deviceVT, (outputVT->n)*(outputVT->n)*sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(hostS, deviceS, (outputS->n)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&infoGpu, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost);

    //Save SVD
    output = {outputU, outputS, outputVT};

    //Cleaning cuda memory
    cudaFree(deviceA);
    cudaFree(deviceU);
    cudaFree(deviceVT);
    cudaFree(deviceS);
    cudaFree(deviceWork);
    cusolverDnDestroy(cusolverH);

    cudaDeviceReset();

    return output;
}