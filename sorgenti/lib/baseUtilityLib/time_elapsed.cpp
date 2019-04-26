#include <baseUtilityLib.h>

using namespace baseUtl;

//*************************************************
//  Compute total runTime = init + work + finalize 
//************************************************

int64_t TimeElapsed::getTotalTime(){
    
    return init + working + finalize;
}