#ifndef __LAB1_GEMM_UTILS_H__
#define __LAB1_GEMM_UTILS_H__

namespace gemm::utils
{
    void printMatrix(const float *matrix, const int M, const int N, const int pretty = 8);

    void randomFillMatrix(float *matrix, const int M, const int N, const float a = 0.0, const float b = 1.0);

    void oneFillMatrix(float *matrix, const int M, const int N);

    bool checkSameMatrix(const float *expected, const float *got, const int M, const int N);

} // namespace gemm::utils

#endif // __LAB1_GEMM_UTILS_H__