#include <svdLib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace svd;

CuSolverGeSvdJ::CuSolverGeSvdJ(){}

//*******************************************************************************************************
//  Save the vector on which SVD will be executed, create cuSolver additional data and move data on HOST
//  input:  + matrix (Matrix*) float, collum-major
//******************************************************************************************************
void CuSolverGeSvdJ::init(Matrix* matrix){

    //Call parent method
    SvdCudaEngine::init(matrix);

    //Configuration gesvdj
    cusolverDnCreateGesvdjInfo(&gesvdjParams);
    //cusolverDnXgesvdjSetTolerance(gesvdjParams, tolerance);
    //cusolverDnXgesvdjSetMaxSweeps(gesvdjParams, maxSweeps);

    //Allocate Space on device
    cusolverDnSgesvdj_bufferSize(
        cusolverH, 
        jobZ,
        econ,
        input->m,
        input->n,
        deviceA,
        input->ld,
        deviceS,
        deviceU,
        input->m,
        deviceVT,
        input->n,
        &lWork,
        gesvdjParams);
        
    cudaMalloc((void**) &deviceWork , sizeof(double)*lWork);
}

//*********************************************
//  CuSolver SVD decomposition (JACOBI METHOD)
//********************************************
void CuSolverGeSvdJ::work(){

    //DGESVDJ
    cusolverDnSgesvdj(
        cusolverH,
        jobZ,
        econ,
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
        deviceInfo,
        gesvdjParams
    );
    cudaDeviceSynchronize();
    printStat();
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
//  Print additional CuSolver stats
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
void CuSolverGeSvdJ::printStat(){

    double residual = 0;
    int executedSweeps = 0;

    cusolverDnXgesvdjGetSweeps(
        cusolverH,
        gesvdjParams,
        &executedSweeps);
    
    cusolverDnXgesvdjGetResidual(
        cusolverH,
        gesvdjParams,
        &residual);

    std::cout<<"Residual |A - U*S*V**H|_F = "<<residual;
    std::cout<<"\nNumber of executed sweeps = "<<executedSweeps<<std::endl;
}