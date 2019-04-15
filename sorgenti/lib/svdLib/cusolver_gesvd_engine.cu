#include <svdLib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace svd;

CuSolverGeSvd::CuSolverGeSvd(){}

//*******************************************************************************************************
//  Save the vector on which SVD will be executed, create cuSolver additional data and move data on HOST
//  input:  + matrix (Matrix*) float, collum-major
//******************************************************************************************************
void CuSolverGeSvd::init(Matrix* matrix){

    //Call parent method
    SvdCudaEngine::init(matrix);

    //Allocate WorkSpace on device
    cusolverDnSgesvd_bufferSize(cusolverH, matrix->m, matrix->n, &lWork);
    cudaMalloc((void**) &deviceWork , sizeof(double)*lWork);
    
}

//*****************************************
//  CuSOlver SVD decomposition (QR METHOD)
//****************************************
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

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
thrust::host_vector<Matrix*> CuSolverGeSvd::getOutputMatrices(){
    
    cudaFree(deviceInfo);
    if(deviceRWork != NULL )
        cudaFree(deviceRWork);
    return SvdCudaEngine::getOutputMatrices(); 
}