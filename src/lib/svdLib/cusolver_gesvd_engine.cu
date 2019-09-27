#include <svdLib.h>

using namespace svd;
using namespace baseUtl;

CuSolverGeSvd::CuSolverGeSvd(){}

//*********************************************************************************************************
//  Save the vector on which SVD will be executed, create cuSolver additional data and move data on DEVICE
//  input:  + matrix (Matrix*) float, collum-major
//********************************************************************************************************
void CuSolverGeSvd::init(Matrix* matrix){

    //Call parent method
    SvdCudaEngine::init(matrix);

    //Allocate WorkSpace on device
    cusolverDnSgesvd_bufferSize(*cusolverH, matrix->m, matrix->n, &lWork);
    cudaMalloc((void**) &deviceWork , sizeof(double)*lWork);
    
}

//*****************************************
//  CuSOlver SVD decomposition (QR METHOD)
//****************************************
void CuSolverGeSvd::work(){

    //DGESVD
    cusolverDnSgesvd(
        *cusolverH,
        'A',
        'A',
        input->m,
        input->n,
        raw_pointer_cast(deviceA->data()),
        input->ld,
        raw_pointer_cast(deviceS->data()),
        raw_pointer_cast(deviceU->data()),
        input->m,
        raw_pointer_cast(deviceVT->data()),
        input->n,
        deviceWork,
        lWork,
        deviceRWork,
        deviceInfo
    );
    cudaDeviceSynchronize();
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major HOST
//*****************************************************************
thrust::host_vector<Matrix*> CuSolverGeSvd::getOutputMatrices(){
    
    cudaFree(deviceInfo);
    if(deviceRWork != NULL )
        cudaFree(deviceRWork);
    return SvdCudaEngine::getOutputMatrices(); 
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major DEViCE
//*****************************************************************
thrust::host_vector<baseUtl::Matrix*> CuSolverGeSvd::getDeviceOutputMatrices(){

    cudaFree(deviceInfo);
    if(deviceRWork != NULL )
        cudaFree(deviceRWork);
    return SvdCudaEngine::getDeviceOutputMatrices(); 
}