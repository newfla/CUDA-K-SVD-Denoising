#include <svdLib.h>

using namespace svd;
using namespace baseUtl;

//***************************************************************************
//  Constructor   
//  input:  + svdEngine (SvdEngine*) lifetime associated with this container 
//**************************************************************************
SvdContainer::SvdContainer(SvdEngine* svdEngine){

    this->svdEngine = svdEngine;
    this->timeElapsed = new TimeElapsed();
}

//****************************************
//  Destructor
//  Free resources acquired (HOST/DEVICE) 
//***************************************
SvdContainer::~SvdContainer(){

    if(timeElapsed!=NULL)
        delete timeElapsed;
        
    if(svdEngine!=NULL)
        delete svdEngine;
}

//********************************************************
//  Save the matrix* and measure SvdEngine init overhead   
//  input:  + matrix (Matrix*) float, collum-major
//*******************************************************
void SvdContainer::setMatrix(Matrix* matrix){

    auto start = std::chrono::steady_clock::now();
    svdEngine->init(matrix);
    auto end = std::chrono::steady_clock::now();
    timeElapsed->init = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
}

//**********************************************************************************
//  Obtain input matrix SVD decompisition and measure SvdEngine last phase overhead 
//  output:  + matrices (Matrix*) float, collum-major HOST
//*********************************************************************************
thrust::host_vector<Matrix*> SvdContainer::getOutputMatrices(){

    thrust::host_vector<Matrix*> output;

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

//**********************************************************************************
//  Obtain input matrix SVD decompisition and measure SvdEngine last phase overhead 
//  output:  + matrices (Matrix*) float, collum-major DEVICE
//*********************************************************************************
thrust::device_vector<Matrix*> SvdContainer::getDeviceOutputMatrices(){

    thrust::host_vector<Matrix*> output;

    auto start = std::chrono::steady_clock::now();
    svdEngine->work();
    auto end = std::chrono::steady_clock::now();
    timeElapsed->working = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    start = std::chrono::steady_clock::now();
    output = svdEngine->getDeviceOutputMatrices();
    end = std::chrono::steady_clock::now();
    timeElapsed->finalize = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    return output;
}

//*********************************************
//  Obtain time stat
//  output:  + timers (TimeElapsed*) ms timers
//********************************************
TimeElapsed* SvdContainer::getTimeElapsed(){
    
    return timeElapsed;
}