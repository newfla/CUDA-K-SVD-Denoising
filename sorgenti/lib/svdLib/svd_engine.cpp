#include <svdLib.h>

using namespace svd;

SvdEngine::SvdEngine(){}

//**************************************
//  Destructor
//  Free Matrix* acquired (HOST/DEVICE) 
//*************************************
SvdEngine::~SvdEngine(){

    delete input;
    for(Matrix* matrix : output)
        delete matrix;
}

//*************************************************
//  Save the vector on which SVD will be executed
//  input:  + matrix (Matrix*) float, collum-major
//************************************************
void SvdEngine::init(Matrix* matrix){

    input = matrix;
}

//************************************************************************
//  SvdEngine Factory method instantiates an object based on type
//  input:  + type (SvdEngineType) of SvdEngine that will be used for SVD 
//  output: + engine (SvdEngine*)
//***********************************************************************
SvdEngine* SvdEngine::factory(SvdEngineType type){
    
    switch (type)
    {
        case CUSOLVER_GESVD:
            return new CuSolverGeSvd();

        case CUSOLVER_GESVDJ:
            return new CuSolverGeSvdJ();
            
        default:
            return NULL;
    }
}
