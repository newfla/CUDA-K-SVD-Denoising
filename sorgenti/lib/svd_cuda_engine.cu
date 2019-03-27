#include <svdLib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace svd;

void SvdCudaEngine::init(Matrix* matrix){

    //Call parent method
    SvdEngine::init(matrix);

    //Create cusolverDn handle
    cusolverDnCreate(&cusolverH) ;
    
    //Save matrix mem dimension
    size_t space = (matrix->ld)*(matrix->n)*sizeof(double);

    //Allocate memory on device
    cudaMalloc((void**) &deviceA, space);
    cudaMalloc((void**) &deviceU, (matrix->m)*(matrix->m)*sizeof(double));
    cudaMalloc((void**) &deviceS, (matrix->n)*sizeof(double));
    cudaMalloc((void**) &deviceVT, (matrix->n)*(matrix->n)*sizeof(double));

    //Copy matrix on device
    cudaMemcpy(deviceA, matrix->matrix, space, cudaMemcpyHostToDevice);

} 

std::vector<Matrix*> SvdCudaEngine::getOutputMatrices(){

    double *hostU, *hostVT, *hostS;
    Matrix *outputU, *outputVT, *outputS;

    //Cpu matrix resource allocation
    hostU = new double[(input->m)*(input->m)]();
    hostVT = new double[(input->n)*(input->n)]();
    hostS = new double[input->n]();

    //Output matrices
    outputU = new Matrix(input->m, input->m, input->m, hostU);
    outputVT = new Matrix(input->n, input->n, input->n, hostVT);
    outputS = new Matrix (1, input->n, input->n, hostS);

    //Copy back to host
    cudaMemcpy(hostU, deviceU, (outputU->ld)*(outputU->n)*sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(hostVT, deviceVT, (outputVT->ld)*(outputVT->n)*sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(hostS, deviceS, (outputS->ld)*sizeof(double), cudaMemcpyDeviceToHost);

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