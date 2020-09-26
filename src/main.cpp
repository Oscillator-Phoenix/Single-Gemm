#include "gemm.h"       // gemm namespace
#include "gemm_utils.h" // randomFillMatrix, printMatrix
#include "sparseCSR.h"  // sparse namespace
#include "use_timer.h"  // ABTMS, ABTME

#include <iostream>
#include <memory>
#include <string>
#include <vector>

void printSplitLine()
{
    std::printf("================================================================================================\n");
}

void printMessageLine(std::string s)
{
    std::printf("============ %s ===============\n", s.c_str());
}

void TestGemm()
{
    // int M = 256;
    // int N = 512;
    // int K = 256;

    // int M = 1024;
    // int N = 512;
    // int K = 2048;

    // int M = 1024;
    // int N = 1024;
    // int K = 1024;

    // int M = 2048;
    // int N = 2048;
    // int K = 2048;

    int M = 4096;
    int N = 4096;
    int K = 4096;

    printSplitLine();
    std::printf("general matrix multiplication: A[%d][%d] * B[%d][%d] = C[%d][%d]\n", M, N, N, K, M, K);
    printSplitLine();

    float *A = new float[M * N];
    float *B = new float[N * K];
    float *CTrival = new float[M * K];
    float *COpt = new float[M * K];
    float *CStrassen = new float[M * K];

    gemm::utils::randomFillMatrix(A, M, N);
    gemm::utils::randomFillMatrix(B, N, K);

    printMessageLine("Used Real Time");

    ABTMS("generalMatMulTrival");
    // gemm::generalMatMulTrival(A, B, CTrival, M, N, K);
    ABTME("generalMatMulTrival");

    ABTMS("generalMatMulOpt");
    gemm::generalMatMulOpt(A, B, COpt, M, N, K);
    ABTME("generalMatMulOpt");
    if (false == gemm::utils::checkSameMatrix(CTrival, COpt, M, K))
    {
        printMessageLine("Wrong Answer: generalMatMulOpt check failed");
    }

    ABTMS("generalMatMulStrassen");
    gemm::generalMatMulStrassen(A, B, CStrassen, M, N, K);
    ABTME("generalMatMulStrassen");
    if (false == gemm::utils::checkSameMatrix(CTrival, CStrassen, M, K))
    {
        printMessageLine("Wrong Answer: generalMatMulStrassen check failed");
    }

    printSplitLine();
    printMessageLine("Matrix A...");
    gemm::utils::printMatrix(A, M, N);
    printMessageLine("Matrix B...");
    gemm::utils::printMatrix(B, N, K);
    printMessageLine("Matrix C...");
    gemm::utils::printMatrix(CTrival, M, K);
    printSplitLine();

    delete[] A;
    delete[] B;
    delete[] CTrival;
    delete[] COpt;
    delete[] CStrassen;

    printMessageLine("done");
}

void TestSparse1()
{
    // array([[1., 9., 0., 0., 0.],
    //        [0., 0., 2., 0., 3.],
    //        [0., 0., 0., 0., 0.],
    //        [0., 0., 0., 4., 0.],
    //        [0., 0., 0., 0., 5.]])
    std::vector<sparse::ElementCOO> array1;
    array1.push_back(sparse::ElementCOO{0, 0, 1.0});
    array1.push_back(sparse::ElementCOO{1, 2, 2.0});
    array1.push_back(sparse::ElementCOO{1, 4, 3.0});
    array1.push_back(sparse::ElementCOO{3, 3, 4.0});
    array1.push_back(sparse::ElementCOO{4, 4, 5.0});
    array1.push_back(sparse::ElementCOO{0, 1, 9.0});

    // array([[1., 0., 0., 0., 0.],
    //        [0., 0., 2., 0., 3.],
    //        [0., 0., 0., 0., 0.],
    //        [6., 0., 0., 4., 0.],
    //        [7., 8., 0., 0., 5.]])
    std::vector<sparse::ElementCOO> array2;
    array2.push_back(sparse::ElementCOO{0, 0, 1.0});
    array2.push_back(sparse::ElementCOO{1, 4, 3.0});
    array2.push_back(sparse::ElementCOO{3, 3, 4.0});
    array2.push_back(sparse::ElementCOO{1, 2, 2.0});
    array2.push_back(sparse::ElementCOO{4, 4, 5.0});
    array2.push_back(sparse::ElementCOO{3, 0, 6.0});
    array2.push_back(sparse::ElementCOO{4, 0, 7.0});
    array2.push_back(sparse::ElementCOO{4, 1, 8.0});

    auto m1 = sparse::SparseCSR(5, 5, array1.size(), array1.data());
    std::cout << m1;

    auto m2 = sparse::SparseCSR(5, 5, array2.size(), array2.data());
    std::cout << m2;

    auto m1t = m1.Transpose();
    std::cout << m1t;

    auto m2t = m2.Transpose();
    std::cout << m2t;

    auto sum = m1.Add(m2);
    std::cout << sum;
}

void TestSparse2()
{
    // array([[10., 0., 5., 7.],
    //        [ 2., 1., 0., 0.],
    //        [ 3., 0., 4., 0.]])
    std::vector<sparse::ElementCOO> array1;
    array1.push_back(sparse::ElementCOO{0, 0, 10.0});
    array1.push_back(sparse::ElementCOO{0, 2, 5.0});
    array1.push_back(sparse::ElementCOO{0, 3, 7.0});
    array1.push_back(sparse::ElementCOO{1, 0, 2.0});
    array1.push_back(sparse::ElementCOO{1, 1, 1.0});
    array1.push_back(sparse::ElementCOO{2, 0, 3.0});
    array1.push_back(sparse::ElementCOO{2, 2, 4.0});
    auto m1 = sparse::SparseCSR(3, 4, array1.size(), array1.data());

    // array([[2.,  0],
    //        [4.,  8.],
    //        [0., 14.],
    //        [3.,  5.]])
    std::vector<sparse::ElementCOO> array2;
    array2.push_back(sparse::ElementCOO{0, 0, 2.0});
    array2.push_back(sparse::ElementCOO{1, 0, 4.0});
    array2.push_back(sparse::ElementCOO{1, 1, 8.0});
    array2.push_back(sparse::ElementCOO{2, 1, 14.0});
    array2.push_back(sparse::ElementCOO{3, 0, 3.0});
    array2.push_back(sparse::ElementCOO{3, 1, 5.0});
    auto m2 = sparse::SparseCSR(4, 2, array2.size(), array2.data());

    std::cout << m1;

    std::cout << m2;

    auto m1t = m1.Transpose();
    std::cout << m1t;

    auto m2t = m2.Transpose();
    std::cout << m2t;

    auto prod = m1.Mul(m2);
    std::cout << prod;
}

int main()
{
    TestGemm();
    // TestSparse1();
    // TestSparse2();

    return 0;
}
