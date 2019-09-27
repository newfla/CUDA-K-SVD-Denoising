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

    for(Denoiser* denoiser : denoisers)
        delete denoiser;
}

//**********************************************
//  Obtain Denoiser List
//  output:  + denoisers (host_vector<Denoiser*>) 
//************************************************
thrust::host_vector<denoising::Denoiser*>BatchDenoiser::getDenoiserList(){

    return denoisers;
}

//***********************************************************************************************
//  BatchDenoiser Factory method instantiates an object based on type
//  input:  + type (DenoiserType) of denoisers that will be used
//          + jsonFile (string) contains info on where load/save images and denoising parameters 
//  output: + batchDenoiser (BatchDenoiser*)
//**********************************************************************************************
BatchDenoiser* BatchDenoiser::factory(std::string jsonFile){
    
    Object config, globalParams, file;
    Array files;
    DenoiserTypeMap map;
    std::string fileName, inputFolder, outputFolder, refFile;
    std::vector<std::string> skip ={"..", "."};
    int globalPatchWidthDim, globalPatchHeightDim, globalSlidingWidth, globalSlidingHeight, globalAtoms, globalIter, globalOmpIter, globalMinOmpIterBatch, globalSubImageWidthDim, globalSubImageHeightDim;
    bool globalSpeckle, globalBW;
    std::string globalType;

    int patchWidthDim, patchHeightDim, slidingWidth, slidingHeight, atoms, iter, ompIter, minOmpIterBatch, subImageWidthDim, subImageHeightDim;
    bool speckle, bw;
    std::string type;

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
    globalBW = globalParams.get<bool> ("B&W");
    globalType = globalParams.get<std::string> ("type");

    globalAtoms = (int) globalParams.get<Number> ("atoms");
    globalIter = (int) globalParams.get<Number> ("ksvdIter");
    globalOmpIter = (int) globalParams.get<Number> ("ompIter");
    globalMinOmpIterBatch= (int) globalParams.get<Number> ("minOmpIterBatch");
    globalSubImageWidthDim = (int) globalParams.get<Number> ("subImageWidthDim");
    globalSubImageHeightDim = (int) globalParams.get<Number> ("subImageHeightDim");

    BatchDenoiser* batchDenoiser = new BatchDenoiser();

    for(int i = 0; i < files.size(); i++){

        file = files.get<Object>(i);
        patchWidthDim = (int) file.get<Number> ("patchWidthDim", globalPatchWidthDim);
        patchHeightDim = (int) file.get<Number> ("patchHeightDim", globalPatchHeightDim);
        slidingWidth = (int) file.get<Number> ("slidingWidth", globalSlidingWidth);
        slidingHeight = (int) file.get<Number> ("slidingHeight", globalSlidingHeight);
        atoms = (int) file.get<Number> ("atoms", globalAtoms);
        iter = (int) file.get<Number> ("ksvdIter", globalIter);
        ompIter = (int) file.get<Number> ("ompIter", globalOmpIter);
        fileName = (std::string) file.get<std::string>("name");
        refFile = (std::string) file.get<std::string>("ref","");
        speckle = file.get<bool>("speckle",globalSpeckle);
        bw = file.get<bool>("B&W",globalBW);
        minOmpIterBatch = (int) file.get<Number> ("minOmpIterBatch", globalMinOmpIterBatch);
        subImageWidthDim = (int) file.get<Number> ("subImageWidthDim", globalSubImageWidthDim);
        subImageHeightDim = (int) file.get<Number> ("subImageHeightDim", globalSubImageHeightDim);
        type = file.get<std::string> ("type", globalType);
        
        Denoiser* denoiser = Denoiser::factory(map[type], inputFolder + "/" + fileName, outputFolder + "/" + std::to_string(patchWidthDim) + "_" + std::to_string(patchHeightDim) + "_" + fileName);
        
        denoiser->patchWidthDim = patchWidthDim;
        denoiser->patchHeightDim = patchHeightDim;
        denoiser->slidingWidth = slidingWidth;
        denoiser->slidingHeight = slidingHeight;
        denoiser->refImage = inputFolder + "/" + refFile;
        denoiser->ompIter = ompIter;
        denoiser->atoms = atoms;
        denoiser->iter = iter;
        denoiser->speckle = speckle;
        denoiser->bw = bw;
        denoiser->minOmpIterBatch = minOmpIterBatch;
        denoiser->subImageWidthDim = subImageWidthDim;
        denoiser->subImageHeightDim = subImageHeightDim;
        
        batchDenoiser->denoisers.push_back(denoiser);         
    }

    streamJson.close();
    return batchDenoiser;
}