#include <svdLib.h>
#include <cuda.h>
//#include <cuda_runtime.h>

using namespace svd;

void CuSolverDnDgeSvd::init(Matrix* matrix){

    //Call parent method
    SvdEngine::init(matrix);
    
    //Save matrix mem dimension
    size_t space = (matrix->m)*(matrix->n)*sizeof(double);

    //Allocate memory on device
    cudaMalloc((void**) &deviceA, space);

    //Copy matrix on device
    cudaMemcpy(deviceA, matrix->matrix, space, cudaMemcpyHostToDevice);

}

void CuSolverDnDgeSvd::work(){
    cudaFree(deviceA);
}

std::vector<Matrix*> CuSolverDnDgeSvd::getOutputMatrices(){
    return std::vector<Matrix*>();
}