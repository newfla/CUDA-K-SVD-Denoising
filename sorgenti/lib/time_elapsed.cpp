#include <svdLib.h>
using namespace svd;

int64_t TimeElapsed:: getInitTime(){
    return init;
}

int64_t TimeElapsed::getWorkingTime(){
    return working;
}

int64_t TimeElapsed::getFinalizeTime(){
    return finalize;
}

int64_t TimeElapsed::getTotalTime(){
    return init + working + finalize;
}