#include <svdLib.h>

using namespace svd;
using namespace baseUtl;

CuSolverGeSvdJ::CuSolverGeSvdJ(){}

//*********************************************************************************************************
//  Save the vector on which SVD will be executed, create cuSolver additional data and move data on DEVICE
//  input:  + matrix (Matrix*) float, collum-major
//********************************************************************************************************
void CuSolverGeSvdJ::init(Matrix* matrix){

    //Call parent method
    SvdCudaEngine::econ = this->econ;
    SvdCudaEngine::init(matrix);

    //Configuration gesvdj
    cusolverDnCreateGesvdjInfo(&gesvdjParams);
    //cusolverDnXgesvdjSetTolerance(gesvdjParams, tolerance);
    //cusolverDnXgesvdjSetMaxSweeps(gesvdjParams, maxSweeps);

    //Allocate Space on device
    cusolverDnSgesvdj_bufferSize(
        *cusolverH, 
        jobZ,
        econ,
        input->m,
        input->n,
        raw_pointer_cast(deviceA->data()),
        input->ld,
        raw_pointer_cast(deviceS->data()),
        raw_pointer_cast(deviceU->data()),
        input->m,
        raw_pointer_cast(deviceVT->data()),
        input->n,
        &lWork,
        gesvdjParams);
        
    cudaMalloc((void**) &deviceWork , sizeof(float)*lWork);
}

//*********************************************
//  CuSolver SVD decomposition (JACOBI METHOD)
//********************************************
void CuSolverGeSvdJ::work(){

    //DGESVDJ
    cusolverDnSgesvdj(
        *cusolverH,
        jobZ,
        econ,
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
        deviceInfo,
        gesvdjParams
    );
    cudaDeviceSynchronize();
 //   printStat();
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
thrust::host_vector<Matrix*> CuSolverGeSvdJ::getOutputMatrices(){
    
    cudaFree(deviceInfo);
    cusolverDnDestroyGesvdjInfo(gesvdjParams);
    return SvdCudaEngine::getOutputMatrices(); 
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major DEViCE
//*****************************************************************
thrust::host_vector<baseUtl::Matrix*> CuSolverGeSvdJ::getDeviceOutputMatrices(){

    cudaFree(deviceInfo);
    cusolverDnDestroyGesvdjInfo(gesvdjParams);
    return SvdCudaEngine::getDeviceOutputMatrices(); 
}

//******************************************************************
//  Print additional CuSolver stats
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
void CuSolverGeSvdJ::printStat(){

    double residual = 0;
    int executedSweeps = 0;

    cusolverDnXgesvdjGetSweeps(
        *cusolverH,
        gesvdjParams,
        &executedSweeps);
    
    cusolverDnXgesvdjGetResidual(
        *cusolverH,
        gesvdjParams,
        &residual);

    std::cout<<"Residual |A - U*S*V**H|_F = "<<residual;
    std::cout<<"\nNumber of executed sweeps = "<<executedSweeps<<std::endl;
}