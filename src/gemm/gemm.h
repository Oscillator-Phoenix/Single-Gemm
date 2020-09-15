#ifndef __LAB1_GEMM_H__
#define __LAB1_GEMM_H__

namespace gemm
{
    // general matrix addition
    void gemaTrival(const float *A, const float *B, float *C, const int M, const int N);

    // general matrix subtraction
    void gemsTrival(const float *A, const float *B, float *C, const int M, const int N);

    // general matrix multiplication
    void gemmTrival(const float *A, const float *B, float *C, const int M, const int N, const int K);

    // general matrix multiplication
    void gemmStrassen(const float *A, const float *B, float *C, const int M, const int N, const int K);

} // namespace gemm

#endif // __LAB1_GEMM_H__