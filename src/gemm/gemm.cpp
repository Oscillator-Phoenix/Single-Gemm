#include "gemm.h"
#include "gemm_utils.h"

// #include <iostream>

namespace gemm
{

    // Matrix is slice of matrix data.
    struct Matrix
    {
        float *data;
        int M;
        int N;
        int stride;

        Matrix(float *data, const int M, const int N, const int stride)
        {
            this->data = data;
            this->M = M;
            this->N = N;
            this->stride = stride;
        }
    };

    void MatrixMatAdd(const Matrix &A, const Matrix &B, Matrix &C)
    {
        int M = A.M;
        int N = A.N;

        float *a = A.data;
        float *b = B.data;
        float *c = C.data;

        int aStride = A.stride;
        int bStride = B.stride;
        int cStride = C.stride;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                c[i * cStride + j] = a[i * aStride + j] + b[i * bStride + j]; // add
            }
        }
    }

    void MatrixMatSub(const Matrix &A, const Matrix &B, Matrix &C)
    {
        int M = A.M;
        int N = A.N;

        float *a = A.data;
        float *b = B.data;
        float *c = C.data;

        int aStride = A.stride;
        int bStride = B.stride;
        int cStride = C.stride;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                c[i * cStride + j] = a[i * aStride + j] - b[i * bStride + j]; // sub
            }
        }
    }

    void MatrixCopy(Matrix &dest, const Matrix &src)
    {
        int M = dest.M;
        int N = dest.N;

        float *_dest = dest.data;
        float *_src = src.data;

        int destStride = dest.stride;
        int srcStride = src.stride;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                _dest[i * destStride + j] = _src[i * srcStride + j]; // copy
            }
        }
    }

    void MatrixMatMul(const Matrix &A, const Matrix &B, Matrix &C)
    {
        int M = A.M;
        int N = A.N;
        int K = B.N;

        float *a = A.data;
        float *b = B.data;
        float *c = C.data;

        int aStride = A.stride;
        int bStride = B.stride;
        int cStride = C.stride;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < K; j++)
            {
                float dot = 0.0;
                for (int p = 0; p < N; p++)
                {
                    dot += a[i * aStride + p] * b[p * bStride + j];
                }
                c[i * cStride + j] = dot;
            }
        }
    }

    void MatrixMatMulOpt(const Matrix &A, const Matrix &B, Matrix &C)
    {
        // opt: // cycle expand

        int M = A.M;
        int N = A.N;
        int K = B.N;

        float *a = A.data;
        float *c = C.data;
        float *b = B.data;

        int aStride = A.stride;
        int bStride = B.stride;
        int cStride = C.stride;

        // copy b to _b with cols-main memory layout
        float *_b = new float[K * N];
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < N; j++)
            {
                _b[i * N + j] = b[j * bStride + i];
            }
        }

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < K; j++)
            {
                int p = 0;
                float dot = 0.0;

                // for (p = 0; p < N; p++)
                // {
                //     dot += a[i * aStride + p] * _b[j * N + p];
                // }

                for (p = 0; p + 8 < N; p += 8)
                {
                    dot += a[i * aStride + (p + 0)] * _b[j * N + (p + 0)];
                    dot += a[i * aStride + (p + 1)] * _b[j * N + (p + 1)];
                    dot += a[i * aStride + (p + 2)] * _b[j * N + (p + 2)];
                    dot += a[i * aStride + (p + 3)] * _b[j * N + (p + 3)];
                    dot += a[i * aStride + (p + 4)] * _b[j * N + (p + 4)];
                    dot += a[i * aStride + (p + 5)] * _b[j * N + (p + 5)];
                    dot += a[i * aStride + (p + 6)] * _b[j * N + (p + 6)];
                    dot += a[i * aStride + (p + 7)] * _b[j * N + (p + 7)];
                }
                for (; p < N; p++)
                {
                    dot += a[i * aStride + p] * _b[j * N + p];
                }

                c[i * cStride + j] = dot;
            }
        }

        delete[] _b;
    }

    const int DimThresholdStrassen = 128;
    const int ScaleThresholdStrassen = (256 * 256 * 256);
    const int MaxDepthStrassen = 32;

    void MatrixMatMulStrassen(const Matrix &A, const Matrix &B, Matrix &C, int depth)
    {
        int M = A.M;
        int N = A.N;
        int K = B.N;

        // evaluate costs of split
        if (depth >= MaxDepthStrassen || M <= DimThresholdStrassen || N <= DimThresholdStrassen || K <= DimThresholdStrassen || M * N * K <= ScaleThresholdStrassen || !(M % 2 == 0 && N % 2 == 0 && K % 2 == 0))
        {
            MatrixMatMul(A, B, C);
            return;
        }

        int halfM = M / 2;
        int halfN = N / 2;
        int halfK = K / 2;

        const Matrix A11 = Matrix(A.data, halfM, halfN, A.stride);
        const Matrix A12 = Matrix(A.data + halfN, halfM, halfN, A.stride);
        const Matrix A21 = Matrix(A.data + halfM * A.stride, halfM, halfN, A.stride);
        const Matrix A22 = Matrix(A.data + halfM * A.stride + halfN, halfM, halfN, A.stride);

        const Matrix B11 = Matrix(B.data, halfN, halfK, B.stride);
        const Matrix B12 = Matrix(B.data + halfK, halfN, halfK, B.stride);
        const Matrix B21 = Matrix(B.data + halfN * B.stride, halfN, halfK, B.stride);
        const Matrix B22 = Matrix(B.data + halfN * B.stride + halfK, halfN, halfK, B.stride);

        Matrix C11 = Matrix(C.data, halfM, halfK, C.stride);
        Matrix C12 = Matrix(C.data + halfK, halfM, halfK, C.stride);
        Matrix C21 = Matrix(C.data + halfM * C.stride, halfM, halfK, C.stride);
        Matrix C22 = Matrix(C.data + halfM * C.stride + halfK, halfM, halfK, C.stride);

        int aSize = halfM * halfN;
        int bSize = halfN * halfK;
        int cSize = halfM * halfK;

        float *_tmpA = new float[aSize];
        float *_tmpB = new float[bSize];
        Matrix tmpA = Matrix(_tmpA, halfM, halfN, halfN);
        Matrix tmpB = Matrix(_tmpB, halfN, halfK, halfK);

        float *_tmpM1 = new float[cSize];
        float *_tmpM2 = new float[cSize];
        float *_tmpM3 = new float[cSize];
        float *_tmpM4 = new float[cSize];
        float *_tmpM5 = new float[cSize];

        Matrix M1 = Matrix(_tmpM1, halfM, halfK, halfK);
        Matrix M4 = Matrix(_tmpM2, halfM, halfK, halfK);
        Matrix M5 = Matrix(_tmpM3, halfM, halfK, halfK);
        Matrix M7 = Matrix(_tmpM4, halfM, halfK, halfK);
        Matrix M3 = Matrix(_tmpM5, halfM, halfK, halfK);

        {
            // M1 = (A11 + A22) (B11 + B22)
            MatrixMatAdd(A11, A22, tmpA);
            MatrixMatAdd(B11, B22, tmpB);
            MatrixMatMulStrassen(tmpA, tmpB, M1, depth + 1);
        }
        {
            // M4 = A22 (B21 – B11)
            MatrixMatSub(B21, B11, tmpB);
            MatrixMatMulStrassen(A22, tmpB, M4, depth + 1);
        }
        {
            // M5 = (A11 + A12) B22
            MatrixMatAdd(A11, A12, tmpA);
            MatrixMatMulStrassen(tmpA, B22, M5, depth + 1);
        }
        {
            // M7 = (A12 – A22) (B21 + B22)
            MatrixMatSub(A12, A22, tmpA);
            MatrixMatAdd(B21, B22, tmpB);
            MatrixMatMulStrassen(tmpA, tmpB, M7, depth + 1);
        }
        {
            // M3 = A11 (B12 – B22)
            MatrixMatSub(B12, B22, tmpB);
            MatrixMatMulStrassen(A11, tmpB, M3, depth + 1);
        }

        {
            // C11 = M1 + M4 – M5 + M7
            MatrixMatAdd(M1, M4, C11);
            MatrixMatSub(C11, M5, C11);
            MatrixMatAdd(C11, M7, C11);
        }
        {
            // C12 = M3 + M5
            MatrixMatAdd(M3, M5, C12);
        }

        Matrix M2 = Matrix(_tmpM3, halfM, halfK, halfK); // _tmpM3 buffer user: M5 -> M2
        Matrix M6 = Matrix(_tmpM4, halfM, halfK, halfK); // _tmpM4 buffer user: M7 -> M6

        {
            // M2 = (A21 + A22) B11
            MatrixMatAdd(A21, A22, tmpA);
            MatrixMatMulStrassen(tmpA, B11, M2, depth + 1);
        }
        {
            // M6 = (A21 – A11) (B11 + B12)
            MatrixMatSub(A21, A11, tmpA);
            MatrixMatAdd(B11, B12, tmpB);
            MatrixMatMulStrassen(tmpA, tmpB, M6, depth + 1);
        }

        {
            // C21 = M2 + M4
            MatrixMatAdd(M2, M4, C21);
        }
        {
            // C22 = M1 – M2 + M3 + M6
            MatrixMatSub(M1, M2, C22);
            MatrixMatAdd(C22, M3, C22);
            MatrixMatAdd(C22, M6, C22);
        }

        delete[] _tmpA;
        delete[] _tmpB;
        delete[] _tmpM1;
        delete[] _tmpM2;
        delete[] _tmpM3;
        delete[] _tmpM4;
        delete[] _tmpM5;
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------------
    // I am split line.
    // -------------------------------------------------------------------------------------------------------------------------------------------------

    void generalMatAdd(const float *A, const float *B, float *C, const int M, const int N)
    {
        const Matrix mA = Matrix((float *)A, M, N, N);
        const Matrix mB = Matrix((float *)B, M, N, N);
        Matrix mC = Matrix(C, M, N, N);

        MatrixMatAdd(mA, mB, mC);
    }

    void generalMatSub(const float *A, const float *B, float *C, const int M, const int N)
    {
        const Matrix mA = Matrix((float *)A, M, N, N);
        const Matrix mB = Matrix((float *)B, M, N, N);
        Matrix mC = Matrix(C, M, N, N);

        MatrixMatSub(mA, mB, mC);
    }

    void generalMatMulTrival(const float *A, const float *B, float *C, const int M, const int N, const int K)
    {
        const Matrix mA = Matrix((float *)A, M, N, N);
        const Matrix mB = Matrix((float *)B, N, K, K);
        Matrix mC = Matrix(C, M, K, K);

        MatrixMatMul(mA, mB, mC);
    }

    void generalMatMulTrivalOpt(const float *A, const float *B, float *C, const int M, const int N, const int K)
    {
        const Matrix mA = Matrix((float *)A, M, N, N);
        const Matrix mB = Matrix((float *)B, N, K, K);
        Matrix mC = Matrix(C, M, K, K);

        MatrixMatMulOpt(mA, mB, mC);
    }

    void generalMatMulStrassen(const float *A, const float *B, float *C, const int M, const int N, const int K)
    {
        const Matrix mA = Matrix((float *)A, M, N, N);
        const Matrix mB = Matrix((float *)B, N, K, K);
        Matrix mC = Matrix(C, M, K, K);

        MatrixMatMulStrassen(mA, mB, mC, 0);
    }

} // namespace gemm