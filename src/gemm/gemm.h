#ifndef __LAB1_GEMM_H__
#define __LAB1_GEMM_H__

namespace gemm
{
    // generalMatAdd is the funciton of general matrix addition
    // input    : A[M][N], B[M][N]
    // function : C = A+B
    // output   : C[M][N]
    void generalMatAdd(const float *A, const float *B, float *C, const int M, const int N);

    // generalMatSub is the funciton of general matrix subtraction
    // input    : A[M][N], B[M][N]
    // function : C = A-B
    // output   : C[M][N]
    void generalMatSub(const float *A, const float *B, float *C, const int M, const int N);

    // generalMatMulTrival is the naive version of general matrix multiplication without optimization.
    // input    : A[M][N], B[N][K]
    // function : C = A*B
    // output   : C[M][K]
    void generalMatMulTrival(const float *A, const float *B, float *C, const int M, const int N, const int K);

    // generalMatMulStrassen implements Strassen Algorithm of general matrix multiplication.
    // input    : A[M][N], B[N][K]
    // function : C = A*B
    // output   : C[M][K]
    void generalMatMulStrassen(const float *A, const float *B, float *C, const int M, const int N, const int K);

} // namespace gemm

#endif // __LAB1_GEMM_H__