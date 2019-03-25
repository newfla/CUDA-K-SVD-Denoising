#include <svdLib.h>
#include <cstdlib>
#include <chrono>

using namespace svd;

SvdContainer::SvdContainer(SvdEngine* svdEngine){
    this->svdEngine = svdEngine;
}

SvdContainer::~SvdContainer(){
    delete svdEngine;
}

void SvdContainer::setMatrix(Matrix* matrix){
    auto start = std::chrono::steady_clock::now();
    svdEngine->init(matrix);
    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

Matrix* SvdContainer::getOutputMatrices(){
    Matrix* output;

    auto start = std::chrono::steady_clock::now();
    svdEngine->work();
    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    start = std::chrono::steady_clock::now();
    output = svdEngine->getOutputMatrices();
    end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return output;
}

TimeElapsed* SvdContainer::getTimeElapsed(){
    return timeElapsed;
}