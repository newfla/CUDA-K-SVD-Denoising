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
    size_t space = (matrix->ld)*(matrix->n)*sizeof(float);

    //Allocate memory on device
    //std::cout<<"risultato cudaMalloc di A: "<<
    cudaMalloc((void**) &deviceA, space);//<<std::endl;
    cudaMalloc((void**) &deviceU, (matrix->ld)*(matrix->m)*sizeof(float));
    cudaMalloc((void**) &deviceS, (matrix->n)*sizeof(float));
    cudaMalloc((void**) &deviceVT, (matrix->n)*(matrix->n)*sizeof(float));

    //Copy matrix on device
    //std::cout<<"risultato memcpy di A: "<<
    cudaMemcpy(deviceA, matrix->matrix, space, cudaMemcpyHostToDevice);//<<std::endl;

} 

std::vector<Matrix*> SvdCudaEngine::getOutputMatrices(){

    float *hostU, *hostVT, *hostS;
    Matrix *outputU, *outputVT, *outputS;

    //Cpu matrix resource allocation
    hostU = new float[(input->m)*(input->m)]();
    hostVT = new float[(input->n)*(input->n)]();
    hostS = new float[input->n]();

    //Allocate memory on host
    outputU = new Matrix(input->ld, input->m, input->m, hostU);
    outputVT = new Matrix(input->n, input->n, input->n, hostVT);
    outputS = new Matrix (1, input->n, 1, hostS);

    //Copy back to host
    cudaMemcpy(hostU, deviceU, (outputU->ld)*(outputU->m)*sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(hostVT, deviceVT, (outputVT->n)*(outputVT->n)*sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(hostS, deviceS, (outputS->n)*sizeof(float), cudaMemcpyDeviceToHost);

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