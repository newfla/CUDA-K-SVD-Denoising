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
        times[0]->init += times[i]->init;
        times[0]->working += times[i]->working;
        times[0]->finalize += times[i]->finalize;
    }
    
    return times;
}

//************************************************************************************************
//  Obtain PSNR stats
//  output:  + psnr (host_vector<host_vector<double>*>) PSNR before/after denoising foreach image
//***********************************************************************************************
thrust::host_vector<thrust::host_vector<double>*> BatchDenoiser::getPsnr(){
    return psnrs;
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

    std::string fileName, inputFolder, outputFolder, refFile;
    std::vector<std::string> skip ={"..", "."};
    int globalPatchWidthDim, globalPatchHeightDim, globalSlidingWidth, globalSlidingHeight, globalAtoms, globalIter, globalOmpIter;
    bool globalSpeckle;
    int patchWidthDim, patchHeightDim, slidingWidth, slidingHeight, atoms, iter, ompIter;
    bool speckle;
    std::fstream streamJson(jsonFile);

    config.parse(streamJson);

    globalParams = config.get<Object>("globalParams");
    inputFolder = config.get<String> ("inputFolder");
    outputFolder = config.get<String> ("outputFolder");
    files = config.get<Array>("files");

    globalPatchWidthDim = (int) globalParams.get<Number> ("patchWidthDim");
    globalPatchHeightDim = (int) globalParams.get<Number> ("patchHeightDim");
    globalSlidingWidth = (int) globalParams.get<Number> ("slidingWidth");
    globalSlidingHeight = (int) globalParams.get<Number> ("slidingHeight");
    globalSpeckle = globalParams.get<bool> ("speckle");

    globalAtoms = (int) globalParams.get<Number> ("atoms");
    globalIter = (int) globalParams.get<Number> ("ksvdIter");
    globalOmpIter = (int) globalParams.get<Number> ("ompIter");

    BatchDenoiser* batchDenoiser = new BatchDenoiser();
    batchDenoiser->times.push_back(new TimeElapsed());

    for(int i = 0; i < files.size(); i++){

        file = files.get<Object>(i);
        patchWidthDim = (int) file.get<Number> ("patchWidthDim", globalPatchWidthDim);
        patchHeightDim = (int) file.get<Number> ("patchHeightDim", globalPatchHeightDim);
        slidingWidth = (int) file.get<Number> ("slidingWidth", globalSlidingWidth);
        slidingHeight = (int) file.get<Number> ("slidingHeight", globalSlidingHeight);
        atoms = (int) file.get<Number> ("atoms", globalAtoms);
        iter = (int) file.get<Number> ("iter", globalIter);
        ompIter = (int) file.get<Number> ("ompIter", globalOmpIter);
        fileName = (std::string) file.get<std::string>("name");
        refFile = (std::string) file.get<std::string>("ref","");
        speckle = file.get<bool>("speckle",globalSpeckle);
        
        Denoiser* denoiser = Denoiser::factory(type, inputFolder + "/" + fileName, outputFolder + "/" + std::to_string(patchWidthDim) + "_" + std::to_string(patchHeightDim) + "_" + fileName);
        
        denoiser->patchWidthDim = patchWidthDim;
        denoiser->patchHeightDim = patchHeightDim;
        denoiser->slidingWidth = slidingWidth;
        denoiser->slidingHeight = slidingHeight;
        denoiser->refImage = inputFolder + "/" + refFile;
        denoiser->ompIter = ompIter;
        denoiser->atoms = atoms;
        denoiser->iter = iter;
        denoiser->speckle = speckle;
        
        batchDenoiser->denoisers.push_back(denoiser);
        batchDenoiser->times.push_back(denoiser->getTimeElapsed());
        batchDenoiser->psnrs.push_back(denoiser->getPsnr());
         
    }

    streamJson.close();
    
    return batchDenoiser;
}