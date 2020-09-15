#ifndef __LAB1_UTILS_H__
#define __LAB1_UTILS_H__

#include <cstdio> // printf
#include <random> // default_random_engine, uniform_real_distribution

void printMatrix(float *matrix, int M, int N, int pretty = 6)
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

void randomFillMatrix(float *matrix, int M, int N, float a = 0.0, float b = 1.0)
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

void oneFillMatrix(float *matrix, int M, int N)
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
    int matrixSize = M * N;
    for (int i = 0; i < matrixSize; i++)
    {
        if (got[i] != expected[i])
        {
            return false;
        }
    }
    return true;
}

#endif //  __LAB1_UTILS_H__