#ifndef __SPARSE_CSR_H__
#define __SPARSE_CSR_H__

#include <iostream>
#include <memory>

namespace sparse
{
    struct ElementCOO
    {
        int row;
        int col;
        float val;
    };

    class SparseCSR
    {
        friend std::ostream &operator<<(std::ostream &out, const sparse::SparseCSR &M);

    private:
        int rows;
        int cols;
        int nnz;

        int *rowStart;
        ElementCOO *array;

    public:
        SparseCSR();
        SparseCSR(const int rows, const int cols, const int nnz);
        SparseCSR(const int rows, const int cols, const int nnz, const ElementCOO *array, const bool isArraySorted = false);

        ~SparseCSR();

        SparseCSR(const SparseCSR &x);
        SparseCSR &operator=(const SparseCSR &x);

        SparseCSR(SparseCSR &&x);
        SparseCSR &operator=(SparseCSR &&x);

        void __sortArray();
        void __buildRowStartFromArray();

        SparseCSR Transpose();
        SparseCSR Add(const SparseCSR &b);
        SparseCSR Sub(const SparseCSR &b);
        SparseCSR Mul(const SparseCSR &b);
    };

} // namespace sparse

#endif // __SPARSE_CSR_H__