# K-SVD CUDA implementation for Image Denoising 

The algorithm uses Orthogonal Matching Pursuit (OMP) for sparsecoding and kSVD for dictionary learning.

The code uses the thrust library to simplify the vectors' management on the device, the cuSolver library to implement the SVD and cuBlas for implementing OMP 

__90% of K-SVD computation runs on GPU__ 

## Dependencies
 
 - Cmake >= 3.0
 - CUDA >= 10.1
 - CImg
 - ImageMagick
 
 Tested on Ubuntu 16.04

## Build
```Shell
cd sorgenti  
mkdir build
cd build   
cmake ..    
make     
```

# Setup data
```Json
"inputFolder" : "absolute path to images ",
    
"outputFolder" : "absolute path where processed images will be saved",
    
"globalParams" : { //for input files
    "patchSquareDim" : 8, //patch sides dimension
    "slidingPatch" : 2, //slidind between patches
    "atoms" : 256, //number of dicitonary elements 
    "iter" : 10, //Ksvd iterations
    "sigma" : 25 //noise variance
},
    
"files": [
    {"name" : "barbara.png", //input image file 
     "ref" : "barbaraRef.png", //no noise image file useful for PSNR
     "sigma": 22 //ovveride sigma
    }
]
```

## Run
```Shell
cd sorgenti/build
./denoising json.config
```
<aside class="warning">
In config.json if outputFolder is the same as inputFolder original images will be overwritten
</aside>

## Class Diagram

### BaseUtilityLib

![UtilityLib Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/BaseUtilityDiagram.png)

### MatUtilityLib

![UtilityLib Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/MatUtilityDiagram.png)

### SvdLib

![SvdLib Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/SvdDiagram.png)

### DenoisingLib

![DenoisingLib Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/DenoisingDiagram.png)

## History
 - Version 0.1 : First alpha
 - Version 0.2 : Improved performance and removed memory leaks

## Issues
 - Supports only square patches
 - Much slower than CPU version



## Credits 
- Based on [trungmanhhuynh](https://github.com/trungmanhhuynh/kSVD-Image-Denoising) CPU version

- [JSON++ library](https://github.com/hjiang/jsonxx)