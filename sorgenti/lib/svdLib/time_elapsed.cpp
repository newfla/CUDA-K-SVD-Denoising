#include <svdLib.h>

using namespace svd;

int64_t TimeElapsed::getTotalTime(){
    return init + working + finalize;
}