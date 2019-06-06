#include <svdLib.h>

using namespace svd;
using namespace baseUtl;
using namespace thrust;

CuSolverGeSvdJBatch::CuSolverGeSvdJBatch(){}

//*********************************************************************************************************
//  Save the vector on which SVD will be executed, create cuSolver additional data and move data on DEVICE
//  input:  + matrix (Matrix*) float, collum-major
//********************************************************************************************************
void CuSolverGeSvdJBatch::init(Matrix* matrix){

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

    deviceU = new device_vector<float>(matrix->ld * matrix->m * matrix->m);
    deviceS = new device_vector<float>(matrix->ld * less);
    deviceVT = new device_vector<float>(matrix->ld * matrix->n * matrix->n);

    //Copy matrix on device
    if(matrix->deviceVector == NULL)
        matrix->copyOnDevice();
    deviceA = matrix->deviceVector;

    //Configuration gesvdj
    cusolverDnCreateGesvdjInfo(&gesvdjParams);
        
}

//*********************************************
//  CuSolver SVD decomposition (JACOBI METHOD)
//********************************************
void CuSolverGeSvdJBatch::work(){
    
    float* aPtr = raw_pointer_cast(deviceA->data());
    float* vPtr = raw_pointer_cast(deviceVT->data());
    float* sPtr = raw_pointer_cast(deviceS->data());
    float* uPtr = raw_pointer_cast(deviceU->data());

    //DGESVDJ
    for (int i = 0; i < input->ld; i++)
    {
        
        //Allocate Space on device
        cusolverDnSgesvdj_bufferSize(*cusolverH, 
                                     jobZ,
                                     econ,
                                     input->m,
                                     input->n,
                                     aPtr + (i * (input->m * input->n)),
                                     input->m,
                                     sPtr + (i * less),
                                     uPtr + (i * (input->m * input->m)),
                                     input->m,
                                     vPtr + (i * (input->n * input->n)),
                                     input->n,
                                     &lWork,
                                     gesvdjParams);

        cudaDeviceSynchronize();
        cudaMalloc((void**) &deviceWork , sizeof(float) * lWork);
    
        cusolverDnSgesvdj(*cusolverH,
                          jobZ,
                          econ,
                          input->m,
                          input->n,
                          aPtr + (i * (input->m * input->n)),
                          input->m,
                          sPtr + (i * less),
                          uPtr + (i * (input->m * input->m)),
                          input->m,
                          vPtr + (i * (input->n * input->n)),
                          input->n,
                          deviceWork,
                          lWork,
                          deviceInfo,
                          gesvdjParams);
        
        cudaDeviceSynchronize();
        cudaFree(deviceWork);
    }
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
thrust::host_vector<Matrix*> CuSolverGeSvdJBatch::getOutputMatrices(){
    
    cudaFree(deviceInfo);
    cusolverDnDestroyGesvdjInfo(gesvdjParams);
    return SvdCudaEngine::getOutputMatrices(); 
}

//******************************************************************
//  Obtain input matrix SVD decompisition and free DEVICE resources 
//  output:  + matrices (Matrix*) float, collum-major DEViCE
//*****************************************************************
thrust::host_vector<baseUtl::Matrix*> CuSolverGeSvdJBatch::getDeviceOutputMatrices(){

    cudaFree(deviceInfo);
    cusolverDnDestroyGesvdjInfo(gesvdjParams);

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

//******************************************************************
//  Print additional CuSolver stats
//  output:  + matrices (Matrix*) float, collum-major
//*****************************************************************
void CuSolverGeSvdJBatch::printStat(){

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