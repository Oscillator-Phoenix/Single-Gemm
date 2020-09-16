#include "gemm.h"       // gemm namespace
#include "gemm_utils.h" // randomFillMatrix, printMatrix
#include "use_timer.h"  // ABTMS, ABTME

#include <iostream>
#include <memory>
#include <string>

void printSplitLine()
{
    std::printf("================================================================================================\n");
}

void printMessageLine(std::string s)
{
    std::printf("============ %s ===============\n", s.c_str());
}

int main()
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    // int M = 2048;
    // int N = 2048;
    // int K = 2048;

    printSplitLine();
    std::printf("general matrix multiplication: A[%d][%d] * B[%d][%d] = C[%d][%d]\n", M, N, N, K, M, K);
    printSplitLine();

    float *A = new float[M * N];
    float *B = new float[N * K];
    float *CTrival = new float[M * K];
    float *COpt = new float[M * K];
    float *CStrassen = new float[M * K];

    gemm::utils::randomFillMatrix(A, M, N);
    gemm::utils::randomFillMatrix(B, N, K);

    printMessageLine("Used Real Time");

    ABTMS("generalMatMulTrival");
    gemm::generalMatMulTrival(A, B, CTrival, M, N, K);
    ABTME("generalMatMulTrival");

    ABTMS("generalMatMulOpt");
    gemm::generalMatMulOpt(A, B, COpt, M, N, K);
    ABTME("generalMatMulOpt");
    if (false == gemm::utils::checkSameMatrix(CTrival, COpt, M, K))
    {
        printMessageLine("Wrong Answer: generalMatMulOpt check failed");
    }

    ABTMS("generalMatMulStrassen");
    gemm::generalMatMulStrassen(A, B, CStrassen, M, N, K);
    ABTME("generalMatMulStrassen");
    if (false == gemm::utils::checkSameMatrix(CTrival, CStrassen, M, K))
    {
        printMessageLine("Wrong Answer: generalMatMulStrassen check failed");
    }

    printSplitLine();
    printMessageLine("Matrix A...");
    gemm::utils::printMatrix(A, M, N);
    printMessageLine("Matrix B...");
    gemm::utils::printMatrix(B, N, K);
    printMessageLine("Matrix C...");
    gemm::utils::printMatrix(CTrival, M, K);
    printSplitLine();

    delete[] A;
    delete[] B;
    delete[] CTrival;
    delete[] COpt;
    delete[] CStrassen;

    printMessageLine("done");
}
