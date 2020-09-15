#ifndef __LAB1_GEMM_H__
#define __LAB1_GEMM_H__

namespace gemm
{
    // general matrix addition
    void generalMatAdd(const float *A, const float *B, float *C, const int M, const int N);

    // general matrix subtraction
    void generalMatSub(const float *A, const float *B, float *C, const int M, const int N);

    // general matrix multiplication, trival algorithm
    void generalMatMulTrival(const float *A, const float *B, float *C, const int M, const int N, const int K);

    // general matrix multiplication, Strassen algorithm
    void generalMatMulStrassen(const float *A, const float *B, float *C, const int M, const int N, const int K);

} // namespace gemm

#endif // __LAB1_GEMM_H__