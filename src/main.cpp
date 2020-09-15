#include "gemm.h"       // gemm namespace
#include "gemm_utils.h" // randomFillMatrix, printMatrix
#include "use_timer.h"  // ABTMS, ABTME

#include <iostream>
#include <memory>

int main()
{
    // int M = 2048;
    // int N = 2048;
    // int K = 2048;

    // int M = 1024;
    // int N = 1024;
    // int K = 1024;

    int M = 512;
    int N = 512;
    int K = 1024;

    // int M = 8;
    // int N = 16;
    // int K = 32;

    // int M = 4;
    // int N = 8;
    // int K = 2;

    float *A = new float[M * N];
    float *B = new float[N * K];
    float *CTrival = new float[M * K];
    float *CStrassen = new float[M * K];

    gemm::utils::randomFillMatrix(A, M, N);
    gemm::utils::randomFillMatrix(B, N, K);

    ABTMS("generalMatMulTrival");
    gemm::generalMatMulTrival(A, B, CTrival, M, N, K);
    ABTME("generalMatMulTrival");

    ABTMS("generalMatMulStrassen");
    gemm::generalMatMulStrassen(A, B, CStrassen, M, N, K);
    ABTME("generalMatMulStrassen");

    if (false == gemm::utils::checkSameMatrix(CTrival, CStrassen, M, K))
    {
        std::cout << "================ Wrong Answer: generalMatMulStrassen check failed =================\n";
    }

    // gemm::utils::printMatrix(A, M, N);
    // gemm::utils::printMatrix(B, N, K);
    // gemm::utils::printMatrix(CTrival, M, K);
    // gemm::utils::printMatrix(CStrassen, M, K);

    delete[] A;
    delete[] B;
    delete[] CTrival;
    delete[] CStrassen;

    std::cout << "done\n";
}