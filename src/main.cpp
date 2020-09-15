#include "gemm.h"      // gemm namespace
#include "use_timer.h" // ABTMS, ABTME
#include "utils.h"     // randomFillMatrix, printMatrix

#include <iostream>
#include <memory>

int main()
{
    int M = 512;
    int N = 512;
    int K = 512;

    float *A = new float[M * N];
    float *B = new float[N * K];
    float *C = new float[M * K];

    randomFillMatrix(A, M, N);
    randomFillMatrix(B, N, K);

    ABTMS("gemmTrival");
    gemm::gemmTrival(A, B, C, M, N, K);
    ABTME("gemmTrival");

    printMatrix(A, M, N);
    printMatrix(B, N, K);
    printMatrix(C, M, K);

    delete[] A;
    delete[] B;
    delete[] C;
}