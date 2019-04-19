#include <utilityLib.h>

using namespace utl;

//*************************************************
//  Compute total runTime = init + work + finalize 
//************************************************

int64_t TimeElapsed::getTotalTime(){
    
    return init + working + finalize;
}