#include <svdLib.h>

using namespace svd;

SvdEngine::SvdEngine(){

}

SvdEngine::~SvdEngine(){
    delete input;
    for(Matrix* matrix : output)
        delete matrix;
}

void SvdEngine::init(Matrix* matrix){
    input = matrix;
}

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
