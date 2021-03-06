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
cd src  
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
    "patchWidthDim" : 12, //patch side dimension
    "patchHeightDim" : 12,  //other patch side dimension
    "slidingWidth" : 3, //sliding between patches x-axis
    "slidingHeight" : 3, //sliding between patches y-axis
    "atoms" : 256, //number of dicitonary elements 
    "ksvditer" : 10, //Ksvd iterations
    "ompIter": 5, //OMP phase limit
    "B&W" : true, //gray scale images
    "speckle" : false, //don't use log/exp transform
    "type": "CUDA_K_GESVDJ", //SVD decomposition alghorithm check denoisingLib.h for further details
},
    
"files": [
    {"name" : "barbara.png", //input image file 
     "ref" : "barbaraRef.png", //no noise image file useful for PSNR
     "patchWidthDim": 22 //override globalPatchWidthDim
    }
]
```

## Run
```Shell
cd src/build
./denoising config.json
```
<aside class="warning">
In config.json if outputFolder is the same as inputFolder original images will be overwritten
</aside>

## Results
| Noise Image | Recovered Image |
| ------------- | ------------- |
| ![image](https://github.com/newfla/Denosing-SVD/blob/master/img/input/barbara.png) | ![image](https://github.com/newfla/Denosing-SVD/blob/master/img/output/barabara.png) |
| ![image](https://github.com/newfla/Denosing-SVD/blob/master/img/input/istanbul2048.jpg) | ![image](https://github.com/newfla/Denosing-SVD/blob/master/img/output/istanbul2048.jpg) |
| ![image](https://github.com/newfla/Denosing-SVD/blob/master/img/input/istanbul.jpg) | ![image](https://github.com/newfla/Denosing-SVD/blob/master/img/output/istanbul.jpg) |

## Activity Diagram

![Overview Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/OverviewDiagram.png)

- [Initialization Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/InitializationDiagram.png)

- [Denoising Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/DenoiseDiagram.png)

- [Build Image Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/BuildImageDenoisedDiagram.png)

## Class Diagram

- [BaseUtilityLib Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/BaseUtilityDiagram.png)

- [MatUtilityLib Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/MatUtilityDiagram.png)

- [SvdLib Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/SvdDiagram.png)

- [DenoisingLib Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/uml/src/DenoisingDiagram.png)

## History
 - Version 0.1 : First alpha

 - Version 0.2 : Improved performance and removed memory leaks

 - Version 0.3 : Improved performance and accuracy. Sigma is no more 
 required. Added support to rectangular patches

 - Version 0.4 : Faster than CPU version

 - Version 0.5 : Fixed Speckle noise support

 - Version 0.6 : Improved API

 - Version 1.0 : General BugFix & improvements. First public version

## Issues
 - I'm sure there are but for now I can't find them



## Credits 
- Based on [npd](https://github.com/npd/ksvd) and [trungmanhhuynh](https://github.com/trungmanhhuynh/kSVD-Image-Denoising) CPU version

- [JSON++ library](https://github.com/hjiang/jsonxx)
