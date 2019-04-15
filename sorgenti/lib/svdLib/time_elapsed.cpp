#include <svdLib.h>

using namespace svd;

//*************************************************
//  Compute total runTime = init + work + finalize 
//************************************************

int64_t TimeElapsed::getTotalTime(){
    
    return init + working + finalize;
}