#include <denoisingLib.h>

using namespace denoising;
using namespace svd;
using namespace baseUtl;
using namespace thrust;
using namespace jsonxx;

BatchDenoiser::BatchDenoiser(){}

//****************************************
//  Destructor
//  Free Denoiser* acquired (HOST/DEVICE) 
//***************************************
BatchDenoiser::~BatchDenoiser(){

    //TimeElapsed[1]...[n] are freed by associated denoiser 
    for(Denoiser* denoiser : denoisers)
        delete denoiser;

}

//************************************************************************
//  Obtain time stats
//  output:  + timers (host_vector<TimeElapsed*>) ms timers foreach image
//***********************************************************************
host_vector<baseUtl::TimeElapsed*> BatchDenoiser::getTimeElapsed(){

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

//***************************************************************************************************************************************************
//  Denoise each image sequentially
//  output:  + status (host_vector<signed char>) foreach image: 0 = done, -1 = image loading failed, -2 = denoising failed, -3 = image saving failed
//**************************************************************************************************************************************************
host_vector<signed char> BatchDenoiser::seqBatchDenoising(){

    host_vector<signed char> results;
    
    for(Denoiser* denoiser : denoisers)
    
        results.push_back(denoiser->denoising());

    return results;
}

//***********************************************************************************************
//  BatchDenoiser Factory method instantiates an object based on type
//  input:  + type (DenoiserType) of denoisers that will be used
//          + jsonFile (string) contains info on where load/save images and denoising parameters 
//  output: + batchDenoiser (BatchDenoiser*)
//**********************************************************************************************
BatchDenoiser* BatchDenoiser::factory(DenoiserType type, std::string jsonFile){
    
    Object config, globalParams, file;
    Array files;
    DIR *dir;
    struct dirent *ent;

    std::string fileName, inputFolder, outputFolder;
    std::vector<std::string> skip ={"..", "."};
    int globalPatchSquareDim, globalSlidingPatch, globalAtoms, globalIter, globalSigma;
    int patchSquareDim, slidingPatch, atoms, iter, sigma;
    std::fstream streamJson(jsonFile);

    config.parse(streamJson);

    globalParams = config.get<Object>("globalParams");
    inputFolder = config.get<String> ("inputFolder");
    outputFolder = config.get<String> ("outputFolder");
    files = config.get<Array>("files");
    globalPatchSquareDim = (int) globalParams.get<Number> ("patchSquareDim");
    globalSlidingPatch = (int) globalParams.get<Number> ("slidingPatch");
    globalAtoms = (int) globalParams.get<Number> ("atoms");
    globalIter = (int) globalParams.get<Number> ("iter");
    globalSigma = (int) globalParams.get<Number> ("sigma");

    BatchDenoiser* batchDenoiser = new BatchDenoiser();
    batchDenoiser->times.push_back(new TimeElapsed());

    for(int i = 0; i < files.size(); i++){

        file = files.get<Object>(i);
        patchSquareDim = (int) file.get<Number> ("patchSquareDim", globalPatchSquareDim);
        slidingPatch = (int) file.get<Number> ("slidingPatch", globalSlidingPatch);
        atoms = (int) file.get<Number> ("atoms", globalAtoms);
        iter = (int) file.get<Number> ("iter", globalIter);
        sigma = (int) file.get<Number> ("sigma", globalSigma);
        fileName = (std::string) file.get<std::string>("name");
        
        Denoiser* denoiser = Denoiser::factory(type, inputFolder +"/"+ fileName, outputFolder +"/"+ fileName);
        
        denoiser->patchSquareDim = patchSquareDim;
        denoiser->slidingPatch = slidingPatch;
    
        CudaKSvdDenoiser* cudaDenoiser = (CudaKSvdDenoiser*) denoiser;
        cudaDenoiser->atoms = atoms;
        cudaDenoiser->iter = iter;
        cudaDenoiser->sigma = sigma;
        
        batchDenoiser->denoisers.push_back(denoiser);
        batchDenoiser->times.push_back(denoiser->getTimeElapsed());   
    }

    streamJson.close();
    
    return batchDenoiser;
}