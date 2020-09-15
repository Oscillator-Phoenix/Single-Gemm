#include "gemm.h"

namespace gemm
{
    // gemd is the funciton of general matrix addition
    // input    : A[M][N], B[M][N]
    // function : C = A+B
    // output   : C[M][N]
    void gemaTrival(const float *A, const float *B, float *C, const int M, const int N)
    {
        int stride = N;
        _gema(A, B, C, M, N, stride, stride, stride);
    }

    // gems is the funciton of general matrix subtraction
    // input    : A[M][N], B[M][N]
    // function : C = A-B
    // output   : C[M][N]
    void gemsTrival(const float *A, const float *B, float *C, const int M, const int N)
    {
        int stride = N;
        _gems(A, B, C, M, N, stride, stride, stride);
    }

    void _gema(const float *A, const float *B, float *C,
               const int M, const int N,
               const int aStride, const int bStride, const int cStride)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i * cStride + j] = A[i * aStride + j] + B[i * bStride + j];
            }
        }
    }

    void _gems(const float *A, const float *B, float *C,
               const int M, const int N,
               const int aStride, const int bStride, const int cStride)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i * cStride + j] = A[i * aStride + j] - B[i * bStride + j];
            }
        }
    }

    // gemmTrival is the naive version of general matrix multiplication without optimization.
    // input    : A[M][N], B[N][K]
    // function : C = A*B
    // output   : C[M][K]
    void gemmTrival(const float *A, const float *B, float *C, const int M, const int N, const int K)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < K; j++)
            {
                float dot = 0.0;
                for (int p = 0; p < N; p++)
                {
                    dot += A[i * N + p] * B[p * K + j];
                }
                C[i * K + j] = dot;
            }
        }
    }

    // gemmStrassen implements Strassen Algorithm of general matrix multiplication.
    // input    : A[M][N], B[N][K]
    // function : C = A*B
    // output   : C[M][K]
    void gemmStrassen(const float *A, const float *B, float *C, const int M, const int N, const int K)
    {
        int depthRemain = M / 512;
        _gemmStrassen(A, B, C, M, N, K, depthRemain);
    }

    void _gemmStrassen(const float *A, const float *B, float *C, const int M, const int N, const int K, const int depthRemain)
    {
        if (depthRemain == 0)
        {
            gemmTrival(A, B, C, M, N, K);
            return;
        }

        int hM = M / 2;
        int hN = N / 2;
        int hK = K / 2;

        // [M/2][N/2]
        const float *A11 = A;
        const float *A12 = A12 + hN;
        const float *A21 = A + hM * N;
        const float *A22 = A21 + hN;

        // [N/2][K/2]
        const float *B11 = B;
        const float *B12 = B11 + hK;
        const float *B21 = B + hN * K;
        const float *B22 = B21 + hK;

        // [M/2][K/2]
        const float *C11 = C;
        const float *C12 = C11 + hK;
        const float *C21 = C + hM * K;
        const float *C22 = C21 + hK;

        float *S1 = new float;
        float *S2;
        float *S3;
        float *S4;
        float *S5;
        float *S6;
        float *S7;
        float *S8;
        float *S9;
        float *S10;

        // S1 = B12 - B22  _
        _gems(B12, B22, );
        // S2 = A11 + A12
        _gema(A11, A12);
        // S3 = A21 + A22
        _gema(A21, A22);
        //S4 = B21 - B11
        _gems(B21, B11);
        //S5 = A11 + A22
        _gema(A11, A22);
        // S6 = B11 + B22
        _gema(B11, B22);
        // S7 = A12 - A22
        _gems(A12, A22);
        // S8 = B21 + B22
        _gema(B21, B22);
        // S9 = A11 - A21
        _gems(A11, A21);
        // S10 = B11 + B12
        _gema(B11, B12);

        float *P1;
        float *P2;
        float *P3;
        float *P4;
        float *P5;
        float *P6;
        float *P7;

        _gemmStrassen(A11, S1, P1, , depthRemain - 1);
        _gemmStrassen(S2, B22, P2, , depthRemain - 1);
        _gemmStrassen(S3, B11, P3, , depthRemain - 1);
        _gemmStrassen(A22, S4, P4, , depthRemain - 1);
        _gemmStrassen(S5, S6, P5, , depthRemain - 1);
        _gemmStrassen(S7, S8, P6, , depthRemain - 1);
        _gemmStrassen(S9, S10, P7, , depthRemain - 1);
    }

} // namespace gemm