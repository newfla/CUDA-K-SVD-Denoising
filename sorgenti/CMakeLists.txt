cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

include_directories("${CMAKE_SOURCE_DIR}/lib/utilityLib")
include_directories("${CMAKE_SOURCE_DIR}/lib/svdLib")
include_directories("${CMAKE_SOURCE_DIR}/lib/denoisingLib")
include_directories("${CMAKE_SOURCE_DIR}/lib/jsonLib")

IF(${CMAKE_VERSION} VERSION_LESS "3.8.0")

    include_directories(SYSTEM "/usr/local/cuda/include")

    project(denoising VERSION 0.1.0)

    find_package(CUDA 9.1 QUIET REQUIRED)

    set(CMAKE_CXX_FLAGS "-std=c++11")

    set(CUDA_64_BIT_DEVICE_CODE ON)
   
    cuda_add_library(jsonLib lib/jsonLib/jsonxx.h lib/jsonLib/jsonxx.cc)

    cuda_add_library(utilityLib lib/utilityLib/utilityLib.h lib/utilityLib/matrix.cpp lib/utilityLib/time_elapsed.cpp lib/utilityLib/matrix_ops.cpp lib/utilityLib/cublas_matrix_mult.cu lib/utilityLib/cublas_matrix_add.cu lib/utilityLib/cublas_matrix_omp.cu)

    cuda_add_library(svdLib lib/svdLib/svdLib.h lib/svdLib/svd_engine.cpp lib/svdLib/svd_container.cpp lib/svdLib/svd_cuda_engine.cu lib/svdLib/cusolver_gesvd_engine.cu lib/svdLib/cusolver_gesvdj_engine.cu)

    cuda_add_library(denoisingLib lib/denoisingLib/denoisingLib.h lib/denoisingLib/denoiser.cpp lib/denoisingLib/batch_denoiser.cpp lib/denoisingLib/cuda_svd_denoiser.cu)

    cuda_add_executable(denoising main.cpp)

    target_link_libraries(denoising denoisingLib svdLib utilityLib jsonLib ${CUDA_cusolver_LIBRARY} ${CUDA_cublas_LIBRARY} X11) 

ELSE()

    set(CMAKE_CUDA_COMPILER /usr/lib/nvidia-cuda-toolkit/bin/nvcc)
    
    IF(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    ENDIF()

    project(denoising VERSION 0.1.0 LANGUAGES CXX CUDA)

    add_library(jsonLib lib/jsonLib/jsonxx.h lib/jsonLib/jsonxx.cc)

    add_library(utilityLib lib/utilityLib/utilityLib.h lib/utilityLib/matrix.cpp lib/utilityLib/time_elapsed.cpp lib/utilityLib/matrix_ops.cpp lib/utilityLib/cublas_matrix_mult.cu lib/utilityLib/cublas_matrix_add.cu lib/utilityLib/cublas_matrix_omp.cu)

    add_library(svdLib lib/svdLib/svdLib.h lib/svdLib/svd_engine.cpp lib/svdLib/svd_container.cpp lib/svdLib/svd_cuda_engine.cu lib/svdLib/cusolver_gesvd_engine.cu lib/svdLib/cusolver_gesvdj_engine.cu)

    add_library(denoisingLib lib/denoisingLib/denoisingLib.h lib/denoisingLib/denoiser.cpp lib/denoisingLib/batch_denoiser.cpp lib/denoisingLib/cuda_svd_denoiser.cu)

    add_executable(denoising main.cpp)

    target_link_libraries(denoising denoisingLib svdLib utilityLib jsonLib cusolver cublas X11)

ENDIF()