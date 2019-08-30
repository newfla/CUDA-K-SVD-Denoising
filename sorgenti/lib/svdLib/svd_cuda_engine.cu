#include <svdLib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace svd;
using namespace baseUtl;
using namespace thrust;

cusolverDnHandle_t* SvdCudaEngine::cusolverH = NULL;

SvdCudaEngine::SvdCudaEngine(){}

//*******************************************************************************************************
//  Save the vector on which SVD will be executed, create cuSolver additional data and move data on HOST
//  input:  + matrix (Matrix*) float, collum-major
//******************************************************************************************************
void SvdCudaEngine::init(Matrix* matrix){

    //Call parent method
    SvdEngine::init(matrix);

    //Create cusolverDn handle
    if(cusolverH == NULL){
        cusolverH = new cusolverDnHandle_t();
        cusolverDnCreate(cusolverH);
    }

    //Alocate space for cusolverDnInfo
    cudaMalloc ((void**)&deviceInfo, sizeof(int));

    //Allocate memory on device
    less = matrix->m;
    if(less > matrix->n)
        less = matrix->n;

    deviceU = new device_vector<float>(matrix->ld * matrix->m);
    deviceS = new device_vector<float>(less);
    deviceVT = (!econ) ? new device_vector<float>(matrix->n * matrix->n) : new device_vector<float>(matrix->n * less);

    //Copy matrix on device
    if(matrix->deviceVector == NULL)
        matrix->copyOnDevice();
    deviceA = matrix->deviceVector;
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
thrust::host_vector<Matrix*> SvdCudaEngine::getOutputMatrices(){

    host_vector<float> *hostU, *hostVT, *hostS;
    Matrix *outputU, *outputVT, *outputS;

    //Cpu matrix resource allocation
    hostU = new host_vector<float>(deviceU->begin(), deviceU->end());
    hostVT = new host_vector<float>(deviceVT->begin(), deviceVT->end());
    hostS = new host_vector<float>(deviceS->begin(), deviceS->end());

    //Allocate memory on host
    outputU = new Matrix(input->ld, input->m, input->m, hostU);
    outputS = new Matrix (1, less, 1, hostS);
    outputVT = (!econ) ? new Matrix(input->n, input->n, input->n, hostVT): outputVT = new Matrix(input->n, less, input->n, hostVT);

    //Save SVD
    output.push_back(outputU);
    output.push_back(outputS);
    output.push_back(outputVT);

    //Cleaning cuda memory 
    deviceA->clear();
    deviceA->shrink_to_fit();
    delete deviceA;

    deviceVT->clear();
    deviceVT->shrink_to_fit();
    delete deviceVT;
    
    deviceU->clear();
    deviceU->shrink_to_fit();
    delete deviceU;

    deviceS->clear();
    deviceS->shrink_to_fit();
    delete deviceS;

    cudaFree(deviceWork);
    input->deviceVector = NULL;

    cudaDeviceReset();

    return output;
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major DEVICE
//*****************************************************************
thrust::host_vector<Matrix*> SvdCudaEngine::getDeviceOutputMatrices(){

    Matrix *outputU, *outputVT, *outputS;

    //Allocate memory on host
    outputU = new Matrix(input->ld, input->m, input->m, deviceU);
    outputVT = (!econ) ? new Matrix(input->n, input->n, input->n, deviceVT): outputVT = new Matrix(input->n, less, input->n, deviceVT);
    outputS = new Matrix (1, less, 1, deviceS);
    
    //Save SVD
    output.push_back(outputU);
    output.push_back(outputS);
    output.push_back(outputVT);

    //Cleaning cuda memory 
    cudaFree(deviceWork);
    
    return output;
}

//******************************
// Clear cuSolver additional data
//*****************************
void SvdCudaEngine::finalize(){
    
    if(cusolverH != NULL){
        cusolverDnDestroy(*cusolverH);
        cusolverH = NULL;
    }
}