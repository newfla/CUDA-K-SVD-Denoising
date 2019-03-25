cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

include_directories("${CMAKE_SOURCE_DIR}/lib")

IF(${CMAKE_VERSION} VERSION_LESS "3.8.0")

    project(denoising VERSION 0.1.0)

    find_package(CUDA 9.1 QUIET REQUIRED)

    set(CUDA_64_BIT_DEVICE_CODE ON)

   #cuda_add_executable(denoising main.cu lib/mat_mat_cuda.cu)

ELSE()

    set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc)
    
    IF(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    ENDIF()

    project(denoising VERSION 0.1.0 LANGUAGES CXX CUDA)

    add_library(svdLib lib/svdLib.h lib/matrix.cpp lib/time_elapsed.cpp lib/svd_container.cpp)

    add_executable(denoising main.cu)

ENDIF()