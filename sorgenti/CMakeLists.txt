cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

include_directories("${CMAKE_SOURCE_DIR}/lib")
include_directories(SYSTEM "/usr/local/cuda/include")

IF(${CMAKE_VERSION} VERSION_LESS "3.8.0")

    project(denoising VERSION 0.1.0)

    find_package(CUDA 9.1 QUIET REQUIRED)

    set(CMAKE_CXX_FLAGS "-std=c++11")

    set(CUDA_64_BIT_DEVICE_CODE ON)
    
    cuda_add_library(svdLib lib/svdLib.h lib/matrix.cpp lib/time_elapsed.cpp lib/svd_engine.cpp lib/svd_container.cpp lib/svd_cuda_engine.cu lib/cusolver_gesvd_engine.cu lib/cusolver_gesvdj_engine.cu)

    cuda_add_executable(denoising main.cpp)

    target_link_libraries(denoising svdLib ${CUDA_cusolver_LIBRARY}) 

ELSE()

    set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc)
    
    IF(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    ENDIF()

    project(denoising VERSION 0.1.0 LANGUAGES CXX CUDA)

    add_library(svdLib lib/svdLib.h lib/matrix.cpp lib/time_elapsed.cpp lib/svd_engine.cpp lib/svd_container.cpp lib/svd_cuda_engine.cu lib/cusolver_gesvd_engine.cu lib/cusolver_gesvdj_engine.cu)

    add_executable(denoising main.cpp)

    target_link_libraries(denoising svdLib cusolver)

ENDIF()