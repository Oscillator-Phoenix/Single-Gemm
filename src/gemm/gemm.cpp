#include "gemm.h"
#include "gemm_utils.h"

namespace gemm
{
    // generalMatAdd is the funciton of general matrix addition
    // input    : A[M][N], B[M][N]
    // function : C = A+B
    // output   : C[M][N]
    void generalMatAdd(const float *A, const float *B, float *C, const int M, const int N)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int pos = i * N + j;
                C[pos] = A[pos] + B[pos];
            }
        }
    }

    // generalMatSub is the funciton of general matrix subtraction
    // input    : A[M][N], B[M][N]
    // function : C = A-B
    // output   : C[M][N]
    void generalMatSub(const float *A, const float *B, float *C, const int M, const int N)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int pos = i * N + j;
                C[pos] = A[pos] - B[pos];
            }
        }
    }

    // generalMatMulTrival is the naive version of general matrix multiplication without optimization.
    // input    : A[M][N], B[N][K]
    // function : C = A*B
    // output   : C[M][K]
    void generalMatMulTrival(const float *A, const float *B, float *C, const int M, const int N, const int K)
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

    void matrixCopy(float *dest, const float *src, const int M, const int N, const int destStride, const int srcStride)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                dest[i * destStride + j] = src[i * srcStride + j];
            }
        }
    }

    const int DimThresholdStrassen = 16;
    const int ScaleThresholdStrassen = (256 * 256 * 256);
    const int MaxRecursionDepthStrassen = 32;

    void _generalMatMulStrassen(const float *A, const float *B, float *C, const int M, const int N, const int K, int depth)
    {
        // evaluate costs of split
        if (depth >= MaxRecursionDepthStrassen || M <= DimThresholdStrassen || N <= DimThresholdStrassen || K <= DimThresholdStrassen || M * N * K <= ScaleThresholdStrassen)
        {
            generalMatMulTrival(A, B, C, M, N, K);
            return;
        }

        int halfM = M / 2;
        int halfN = N / 2;
        int halfK = K / 2;

        int aSize = halfM * halfN;
        int bSize = halfN * halfK;
        int cSize = halfM * halfK;

        float *A11 = new float[aSize];
        float *A12 = new float[aSize];
        float *A21 = new float[aSize];
        float *A22 = new float[aSize];

        float *B11 = new float[bSize];
        float *B12 = new float[bSize];
        float *B21 = new float[bSize];
        float *B22 = new float[bSize];

        matrixCopy(A11, A, halfM, halfN, halfN, N);
        matrixCopy(A12, A + halfN, halfM, halfN, halfN, N);
        matrixCopy(A21, A + halfM * N, halfM, halfN, halfN, N);
        matrixCopy(A22, A + halfM * N + halfN, halfM, halfN, halfN, N);

        // utils::printMatrix(A, M, N);
        // utils::printMatrix(A11, halfM, halfN);
        // utils::printMatrix(A12, halfM, halfN);
        // utils::printMatrix(A21, halfM, halfN);
        // utils::printMatrix(A22, halfM, halfN);

        matrixCopy(B11, B, halfN, halfK, halfK, K);
        matrixCopy(B12, B + halfK, halfN, halfK, halfK, K);
        matrixCopy(B21, B + halfN * K, halfN, halfK, halfK, K);
        matrixCopy(B22, B + halfN * K + halfK, halfN, halfK, halfK, K);

        // utils::printMatrix(B, N, K);
        // utils::printMatrix(B11, halfN, halfK);
        // utils::printMatrix(B12, halfN, halfK);
        // utils::printMatrix(B21, halfN, halfK);
        // utils::printMatrix(B22, halfN, halfK);

        float *tmpA = new float[aSize];
        float *tmpB = new float[bSize];
        float *tmpC = new float[cSize];

        float *M1 = new float[cSize];
        float *M2 = new float[cSize];
        float *M3 = new float[cSize];
        float *M4 = new float[cSize];
        float *M5 = new float[cSize];
        float *M6 = new float[cSize];
        float *M7 = new float[cSize];

        {
            // M1 = (A11 + A22) (B11 + B22)
            generalMatAdd(A11, A22, tmpA, halfM, halfN);
            generalMatAdd(B11, B22, tmpB, halfN, halfK);
            _generalMatMulStrassen(tmpA, tmpB, M1, halfM, halfN, halfK, depth + 1);
        }

        {
            // M2 = (A21 + A22) B11
            generalMatAdd(A21, A22, tmpA, halfM, halfN);
            _generalMatMulStrassen(tmpA, B11, M2, halfM, halfN, halfK, depth + 1);
        }

        {
            // M3 = A11 (B12 – B22)
            generalMatSub(B12, B22, tmpB, halfN, halfK);
            _generalMatMulStrassen(A11, tmpB, M3, halfM, halfN, halfK, depth + 1);
        }

        {
            // M4 = A22 (B21 – B11)
            generalMatSub(B21, B11, tmpB, halfN, halfK);
            _generalMatMulStrassen(A22, tmpB, M4, halfM, halfN, halfK, depth + 1);
        }

        {
            // M5 = (A11 + A12) B22
            generalMatAdd(A11, A12, tmpA, halfM, halfN);
            _generalMatMulStrassen(tmpA, B22, M5, halfM, halfN, halfK, depth + 1);
        }

        {
            // M6 = (A21 – A11) (B11 + B12)
            generalMatSub(A21, A11, tmpA, halfM, halfN);
            generalMatAdd(B11, B12, tmpB, halfN, halfK);
            _generalMatMulStrassen(tmpA, tmpB, M6, halfM, halfN, halfK, depth + 1);
        }

        {
            // M7 = (A12 – A22) (B21 + B22)
            generalMatSub(A12, A22, tmpA, halfM, halfN);
            generalMatAdd(B21, B22, tmpB, halfN, halfK);
            _generalMatMulStrassen(tmpA, tmpB, M7, halfM, halfN, halfK, depth + 1);
        }

        {
            // C11 = M1 + M4 – M5 + M7
            generalMatAdd(M1, M4, tmpC, halfM, halfK);
            generalMatSub(tmpC, M5, tmpC, halfM, halfK);
            generalMatAdd(tmpC, M7, tmpC, halfM, halfK);
            matrixCopy(C, tmpC, halfM, halfK, K, halfK);
        }

        {
            // C12 = M3 + M5
            generalMatAdd(M3, M5, tmpC, halfM, halfK);
            matrixCopy(C + halfK, tmpC, halfM, halfK, K, halfK);
        }

        {
            // C21 = M2 + M4
            generalMatAdd(M2, M4, tmpC, halfM, halfK);
            matrixCopy(C + halfM * K, tmpC, halfM, halfK, K, halfK);
        }

        {
            // C22 = M1 – M2 + M3 + M6
            generalMatSub(M1, M2, tmpC, halfM, halfK);
            generalMatAdd(tmpC, M3, tmpC, halfM, halfK);
            generalMatAdd(tmpC, M6, tmpC, halfM, halfK);
            matrixCopy(C + halfM * K + halfK, tmpC, halfM, halfK, K, halfK);
        }

        delete[] A11;
        delete[] A12;
        delete[] A21;
        delete[] A22;

        delete[] B11;
        delete[] B12;
        delete[] B21;
        delete[] B22;

        delete[] tmpA;
        delete[] tmpB;
        delete[] tmpC;

        delete[] M1;
        delete[] M2;
        delete[] M3;
        delete[] M4;
        delete[] M5;
        delete[] M6;
        delete[] M7;
    }

    // generalMatMulStrassen implements Strassen Algorithm of general matrix multiplication.
    // input    : A[M][N], B[N][K]
    // function : C = A*B
    // output   : C[M][K]
    void generalMatMulStrassen(const float *A, const float *B, float *C, const int M, const int N, const int K)
    {
        _generalMatMulStrassen(A, B, C, M, N, K, 0);
    }

} // namespace gemm