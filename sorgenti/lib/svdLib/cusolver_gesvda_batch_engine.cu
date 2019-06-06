#include <svdLib.h>

using namespace svd;
using namespace baseUtl;
using namespace thrust;

CuSolverGeSvdABatch::CuSolverGeSvdABatch(){}

//*********************************************************************************************************
//  Save the vector on which SVD will be executed, create cuSolver additional data and move data on DEVICE
//  input:  + matrix (Matrix*) float, collum-major
//********************************************************************************************************
void CuSolverGeSvdABatch::init(Matrix* matrix){

    //Call parent method
    SvdEngine::init(matrix);

    //Create cusolverDn handle
    if(cusolverH == NULL){
        cusolverH = new cusolverDnHandle_t();
        cusolverDnCreate(cusolverH);
    }

    //Alocate space for cusolverDnInfo
    cudaMalloc ((void**)&deviceInfo, sizeof(int) * matrix->ld);

    //Allocate memory on device
    less = matrix->m;
    if(less > matrix->n)
        less = matrix->n;

    if(deviceU == NULL){
        deviceU = new device_vector<float>(matrix->ld * matrix->m * matrix->m);
        deviceS = new device_vector<float>(matrix->ld * less);
        deviceVT = new device_vector<float>(matrix->ld * matrix->n * matrix->n);
    }
    
    //Copy matrix on device
    if(matrix->deviceVector == NULL)
        matrix->copyOnDevice();
    deviceA = matrix->deviceVector;

    //Allocate Space on device
    cusolverDnSgesvdaStridedBatched_bufferSize(*cusolverH,
                                        jobZ,
                                        less,
                                        input->m,
                                        input->n,
                                        raw_pointer_cast(deviceA->data()),
                                        input->m,
                                        input->m * input->n,
                                        raw_pointer_cast(deviceS->data()),
                                        less,
                                        raw_pointer_cast(deviceU->data()),
                                        input->m,
                                        input->m * input->m,
                                        raw_pointer_cast(deviceVT->data()),
                                        input->n,
                                        input->n * input->n,
                                        &lWork,
                                        input->ld);
                                        
    cudaMalloc((void**) &deviceWork , sizeof(float) * lWork);
        
}

//*********************************************
//  CuSolver SVD decomposition (APPROX METHOD)
//********************************************
void CuSolverGeSvdABatch::work(){
    
    float* aPtr = raw_pointer_cast(deviceA->data());
    float* vPtr = raw_pointer_cast(deviceVT->data());
    float* sPtr = raw_pointer_cast(deviceS->data());
    float* uPtr = raw_pointer_cast(deviceU->data());
    double RnrmF[input->ld];
    //DGESVDA
    cusolverDnSgesvdaStridedBatched(*cusolverH,
                                    jobZ,
                                    less,
                                    input->m,
                                    input->n,
                                    raw_pointer_cast(deviceA->data()),
                                    input->m,
                                    input->m * input->n,
                                    raw_pointer_cast(deviceS->data()),
                                    less,
                                    raw_pointer_cast(deviceU->data()),
                                    input->m,
                                    input->m * input->m,
                                    raw_pointer_cast(deviceVT->data()),
                                    input->n,
                                    input->n * input->n,
                                    deviceWork,
                                    lWork,
                                    deviceInfo,
                                    RnrmF,
                                    input->ld);
    cudaDeviceSynchronize();
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
thrust::host_vector<Matrix*> CuSolverGeSvdABatch::getOutputMatrices(){
    
    cudaFree(deviceInfo);
    return SvdCudaEngine::getOutputMatrices(); 
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major DEViCE
//*****************************************************************
thrust::host_vector<baseUtl::Matrix*> CuSolverGeSvdABatch::getDeviceOutputMatrices(){

    cudaFree(deviceInfo);
    cudaFree(deviceWork);

    Matrix *outputU, *outputVT, *outputS;

    //Allocate memory on host
    outputU = new Matrix(input->m, input->m, input->m * input->m, deviceU);
    outputVT = new Matrix(input->n, input->n, input->n * input->n, deviceVT);
    outputS = new Matrix (1, less, less, deviceS);
    
    //Save SVD
    output.push_back(outputU);
    output.push_back(outputS);
    output.push_back(outputVT);

    
    return output; 
}