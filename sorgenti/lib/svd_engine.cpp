#include <svdLib.h>

using namespace svd;

SvdEngine::~SvdEngine(){
    std::cout<<"\nengine";
    delete input;
    for(Matrix* matrix : *output)
      delete matrix;
}

void SvdEngine::init(Matrix* matrix){
    input = matrix;
}

SvdEngine* SvdEngine::factory(SvdEngineType type){
    switch (type)
    {
        case CUSOLVER_DN_DGESVD:
            return new CuSolverDnDgeSvd();
            
        default:
            return NULL;
    }
}
