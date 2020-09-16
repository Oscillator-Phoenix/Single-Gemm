CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(gemm)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


ADD_LIBRARY(${PROJECT_NAME} 
            gemm.h 
            gemm.cpp
            gemm_utils.h
            gemm_utils.cpp)
                 


target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -O3
            >)