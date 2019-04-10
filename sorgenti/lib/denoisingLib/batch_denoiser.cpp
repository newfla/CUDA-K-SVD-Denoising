#include <denoisingLib.h>

using namespace denoising;
using namespace svd;

BatchDenoiser::BatchDenoiser(){}

BatchDenoiser::~BatchDenoiser(){


    //Ogni TimeElapsed[1]...[n] viene già de-allocato dai singoli denoiser 
    for(Denoiser* denoiser : denoisers)
        delete denoiser;

}

std::vector<svd::TimeElapsed*> BatchDenoiser::getTimeElapsed(){

    times[0]->init = 0;
    times[0]->init = 0;
    times[0]->init = 0;

    for(int i = 1; i < times.size(); i++)
    {
        times[0]->init += times[1]->init;
        times[0]->working += times[1]->working;
        times[0]->finalize += times[1]->finalize;
    }
    
    return times;
}

std::vector<signed char> BatchDenoiser::seqBatchDenoising(){

    std::vector<signed char> results;

    for(Denoiser* denoiser : denoisers)
        results.push_back(denoiser->denoising());

    return results;
}

BatchDenoiser* BatchDenoiser::factory(DenoiserType type, std::string inputFolder, std::string outputFolder){
    
    DIR *dir;
    struct dirent *ent;

    std::string inputFile, outputFile;
    std::vector<std::string> skip ={"..", "."};

    BatchDenoiser* batchDenoiser = new BatchDenoiser();
    batchDenoiser->times.push_back(new TimeElapsed());

    if ((dir = opendir (inputFolder.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {

            if(std::find(skip.begin(), skip.end(), ent->d_name) != skip.end())
               continue;
        
            inputFile = inputFolder + "/" + ent->d_name;
            outputFile = outputFolder + "/" + ent->d_name;
            
            Denoiser* denoiser = Denoiser::factory(type, inputFile, outputFile);

            batchDenoiser->denoisers.push_back(denoiser);
            batchDenoiser->times.push_back(denoiser->getTimeElapsed());            
        }
        closedir (dir);
    } else
        return NULL;
    return batchDenoiser;
}