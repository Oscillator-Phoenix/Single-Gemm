CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(gemm)

ADD_LIBRARY(${PROJECT_NAME} 
            gemm.h 
            gemm.cpp)
                 
