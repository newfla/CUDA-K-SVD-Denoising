#include <svdLib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace svd;
using namespace baseUtl;
using namespace thrust;

SvdCudaEngine::SvdCudaEngine(){}

//*******************************************************************************************************
//  Save the vector on which SVD will be executed, create cuSolver additional data and move data on HOST
//  input:  + matrix (Matrix*) float, collum-major
//******************************************************************************************************
void SvdCudaEngine::init(Matrix* matrix){

    //Call parent method
    SvdEngine::init(matrix);

    //Create cusolverDn handle
    cusolverDnCreate(&cusolverH) ;

    //Alocate space for cusolverDnInfo
    cudaMalloc ((void**)&deviceInfo, sizeof(int));

    //Allocate memory on device
    less = matrix->m;
    if(less > matrix->n)
        less= matrix->n;

    cudaMalloc((void**) &deviceU, (matrix->ld)*(matrix->m)*sizeof(float));
    cudaMalloc((void**) &deviceS, (less)*sizeof(float));
    cudaMalloc((void**) &deviceVT, (matrix->n)*(matrix->n)*sizeof(float));

    //Copy matrix on device
    if(matrix->deviceVector == NULL)
        matrix->deviceVector = new device_vector<float>(matrix->hostVector->begin(), matrix->hostVector->end());
    deviceA = raw_pointer_cast(matrix->deviceVector->data());
}


//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
thrust::host_vector<Matrix*> SvdCudaEngine::getOutputMatrices(){

    float *hostU, *hostVT, *hostS;
    Matrix *outputU, *outputVT, *outputS;

    //Cpu matrix resource allocation
    hostU = new float[(input->m)*(input->m)]();
    hostVT = new float[(input->n)*(input->n)]();
    hostS = new float[input->n]();

    //Copy back to host
    cudaMemcpy(hostU, deviceU, (input->ld)*(input->m)*sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(hostVT, deviceVT, (input->n)*(input->n)*sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(hostS, deviceS, (less)*sizeof(float), cudaMemcpyDeviceToHost);

    //Allocate memory on host
    outputU = new Matrix(input->ld, input->m, input->m, hostU);
    outputVT = new Matrix(input->n, input->n, input->n, hostVT);
    outputS = new Matrix (1, less, 1, hostS);

    //Save SVD
    output.push_back(outputU);
    output.push_back(outputS);
    output.push_back(outputVT);

    //Cleaning cuda memory 
    cudaFree(deviceA);
    cudaFree(deviceU);
    cudaFree(deviceVT);
    cudaFree(deviceS);
    cudaFree(deviceWork);
    cusolverDnDestroy(cusolverH);
    input->deviceVector = NULL;

    cudaDeviceReset();

    return output;
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major DEVICE
//*****************************************************************
thrust::device_vector<Matrix*> SvdCudaEngine::getDeviceOutputMatrices(){

    Matrix *outputU, *outputVT, *outputS;

    //Wrap raw pointer
    device_ptr<float> u(deviceU),
                    vt(deviceVT),
                    s(deviceS);

    //Allocate memory on host
    outputU = new Matrix(input->ld, input->m, input->m, new device_vector<float>(u, u + (input->ld * input->m)));
    outputVT = new Matrix(input->n, input->n, input->n, new device_vector<float>(vt, vt + (input->n * input->n)));
    outputS = new Matrix (1, input->n, 1, new device_vector<float>(s, s + input->n));

    //Save SVD
    output.push_back(outputU);
    output.push_back(outputS);
    output.push_back(outputVT);

    //Cleaning cuda memory 
    cudaFree(deviceA);
    cudaFree(deviceWork);
    cusolverDnDestroy(cusolverH);
    input->deviceVector = NULL;

    cudaDeviceReset();

    return output;
}