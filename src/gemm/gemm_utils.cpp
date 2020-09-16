#include "gemm_utils.h"

#include <cmath>    // fabs
#include <cstdio>   // printf
#include <iostream> // cout
#include <random>   // default_random_engine, uniform_real_distribution

namespace gemm::utils
{
    void printMatrix(const float *matrix, const int M, const int N, const int pretty)
    {
        int m = (M > pretty) ? pretty : M;
        int n = (N > pretty) ? pretty : N;

        std::printf("\n");

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::printf("%06.4f ", matrix[i * N + j]);
            }
            std::printf("\n");
        }

        std::printf("\n");
    }

    void randomFillMatrix(float *matrix, const int M, const int N, const float a, const float b)
    {
        std::default_random_engine generator;
        generator.seed(std::random_device()());
        std::uniform_real_distribution<float> distribution(a, b);

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix[i * N + j] = distribution(generator);
            }
        }
    }

    void oneFillMatrix(float *matrix, const int M, const int N)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix[i * N + j] = 1.0;
            }
        }
    }

    bool checkSameMatrix(const float *expected, const float *got, const int M, const int N)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int pos = i * N + j;
                float err = fabs(got[pos] - expected[pos]);

                if (err > 1e-1)
                {
                    std::printf("check failed at expcted[%d][%d]=%6.6f, got[%d][%d]=%6.6f\n", i, j, expected[pos], i, j, got[pos]);
                    return false;
                }
            }
        }
        return true;
    }

} // namespace gemm::utils
