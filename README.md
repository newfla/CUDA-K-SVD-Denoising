# K-SVD CUDA implementation for Image Denoising 

The algorithm uses Orthogonal Matching Pursuit (OMP) for sparsecoding and kSVD for dictionary learning.

The code uses the thrust library to simplify the vectors' management on the device, the cuSolver library to implement the SVD and cuBlas for implementing OMP 

__90% of K-SVD computation runs on GPU__ 

## Dependencies
 
 - Cmake >= 3.0
 - CUDA >= 9.1
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
## Run
```Shell
cd sorgenti/build
./denoising noisedPathFolder deNoisedPathFolder
```
<aside class="warning">
If noisedPathFolder is the same as deNoisedPathFolder original images will be overwritten
</aside>

## Class Diagaram
Relationships between classes of different packages are hidden

![Image of Class Diagram](https://github.com/newfla/Denosing-SVD/raw/master/uml/out/classDiagram/diagrammaSVD.png)


## History
 - Version 0.1 : Private Alpha

## Credits
- Based on [trungmanhhuynh](https://github.com/trungmanhhuynh/kSVD-Image-Denoising) CPU version 

- CUDA OMP implementation by [IdanBanani](https://github.com/IdanBanani/Orthogonal-Matching-Pursuit--OMP--and-Batch-OMP-algorithm-)
