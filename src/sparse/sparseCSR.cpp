#include "sparseCSR.h"

#include <algorithm> // std::fill_n, std::copy, std::sort
#include <cstring>   // std::memcpy
#include <utility>   // std::move
#include <vector>    // std::vector

namespace sparse
{
    std::ostream &operator<<(std::ostream &out, const sparse::SparseCSR &M)
    {
        const int pretty = 6;

        out << "\n";
        out << "sparse matrix\n";
        out << "rows = " << M.rows << "\n";
        out << "cols = " << M.cols << "\n";
        out << "nnzs = " << M.nnz << "\n";

        out << "\n";
        int nnz = (pretty * 2 < M.nnz) ? pretty * 2 : M.nnz;
        for (int i = 0; i < nnz; i++)
        {
            ElementCOO coo = M.array[i];
            out << "(" << coo.row << ", " << coo.col << ", " << coo.val << ")\n";
        }

        int rows = (pretty < M.rows) ? pretty : M.rows;
        int cols = (pretty < M.cols) ? pretty : M.cols;

        float *buf = new float[rows * cols];
        std::fill_n(buf, rows * cols, 0.0);

        for (int i = 0; i < M.nnz; i++)
        {
            ElementCOO coo = M.array[i];
            if (coo.row < rows && coo.col < cols)
            {
                buf[coo.row * cols + coo.col] = coo.val;
            }
        }

        out << "\n";
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                out << buf[i * cols + j] << " ";
            }
            out << "\n";
        }
        out << "\n";

        delete[] buf;
    }

    SparseCSR::SparseCSR()
    {
        this->rows = 0;
        this->cols = 0;
        this->nnz = 0;

        this->rowStart = nullptr;
        this->array = nullptr;
    }

    SparseCSR::SparseCSR(const int rows, const int cols, const int nnz)
    {
        this->rows = rows;
        this->cols = cols;
        this->nnz = nnz;

        this->rowStart = new int[rows + 1];
        this->array = new ElementCOO[nnz];
    }

    // isArraySorted = true,  O(nnz)
    // isArraySorted = false, O(nnz * log(nnz))
    SparseCSR::SparseCSR(const int rows, const int cols, const int nnz, const ElementCOO *array, const bool isArraySorted)
    {
        *this = SparseCSR(rows, cols, nnz); // allocate memory

        int *buf = new int[this->rows];

        // buf saves rowSize
        std::fill_n(buf, this->rows, 0);
        for (int i = 0; i < this->nnz; i++)
        {
            buf[array[i].row]++;
        }

        // calculate rowStart
        this->rowStart[0] = 0;
        for (int i = 1; i < this->rows; i++)
        {
            this->rowStart[i] = this->rowStart[i - 1] + buf[i - 1];
        }
        this->rowStart[this->rows] = this->nnz;

        // buf saves rowStart of matrix b
        std::copy_n(this->rowStart, this->rows, buf);

        // calculate array
        for (int i = 0; i < this->nnz; i++)
        {
            int _i = buf[array[i].row];

            this->array[_i].row = array[i].row;
            this->array[_i].col = array[i].col;
            this->array[_i].val = array[i].val;

            buf[array[i].row]++;
        }

        delete[] buf;

        if (isArraySorted == false)
        {
            // expensive cost
            this->__sortArray();
        }
    }

    SparseCSR::~SparseCSR()
    {
        if (this->rowStart != nullptr)
        {
            delete[] rowStart;
        }

        if (this->array != nullptr)
        {
            delete[] array;
        }
    }

    SparseCSR::SparseCSR(const SparseCSR &x)
    {
        *this = SparseCSR(x.rows, x.cols, x.nnz); // allocate memory

        std::copy_n(x.rowStart, x.rows + 1, this->rowStart);
        std::copy_n(x.array, x.nnz, this->array);
    }

    SparseCSR &SparseCSR::operator=(const SparseCSR &x)
    {
        *this = SparseCSR(x.rows, x.cols, x.nnz); // allocate memory

        std::copy_n(x.rowStart, x.rows + 1, this->rowStart);
        std::copy_n(x.array, x.nnz, this->array);

        return *this;
    }

    SparseCSR::SparseCSR(SparseCSR &&x)
    {
        this->rows = x.rows;
        this->cols = x.cols;
        this->nnz = x.nnz;

        this->rowStart = x.rowStart; // move
        this->array = x.array;

        x.rowStart = nullptr;
        x.array = nullptr;
    }

    SparseCSR &SparseCSR::operator=(SparseCSR &&x)
    {
        // prevent from moving myself
        if (&x == this)
        {
            return *this;
        }

        this->rows = x.rows;
        this->cols = x.cols;
        this->nnz = x.nnz;

        this->rowStart = x.rowStart; // move
        this->array = x.array;

        x.rowStart = nullptr;
        x.array = nullptr;

        return *this;
    }

    // O(nnz * log(nnz))
    void SparseCSR::__sortArray()
    {
        std::sort(this->array, this->array + this->nnz, [](const ElementCOO &e1, const ElementCOO &e2) -> bool {
            if (e1.row == e2.row)
            {
                return e1.col < e2.col;
            }
            return e1.row < e2.row;
        });
    }

    // O(nnz)
    void SparseCSR::__buildRowStartFromArray()
    {
        int *buf = new int[this->rows]; // save rowSize
        std::fill_n(buf, this->rows, 0);

        for (int i = 0; i < 0; i++)
        {
            buf[this->array[i].row]++;
        }

        this->rowStart[0] = 0;
        for (int i = 1; i < this->rows; i++)
        {
            this->rowStart[i] = this->rowStart[i - 1] + buf[i - 1];
        }
        this->rowStart[this->rows] = this->nnz;

        delete[] buf;
    }

    // O(nnz)
    SparseCSR SparseCSR::Transpose()
    {
        SparseCSR b(this->cols, this->rows, this->nnz); // allocate memory

        if (nnz > 0)
        {
            int *buf = new int[b.rows];

            // buf saves rowSize of matrix b
            std::fill_n(buf, b.rows, 0);
            for (int i = 0; i < b.nnz; i++)
            {
                buf[this->array[i].col]++;
            }

            // calculate rowStart of matrix b
            b.rowStart[0] = 0;
            for (int i = 1; i < b.rows; i++)
            {
                b.rowStart[i] = b.rowStart[i - 1] + buf[i - 1];
            }
            b.rowStart[b.rows] = b.nnz;

            // buf saves rowStart of matrix b
            std::copy_n(b.rowStart, b.rows, buf);

            // calculate array

            for (int i = 0; i < b.nnz; i++)
            {
                int j = buf[this->array[i].col];

                b.array[j].row = this->array[i].col;
                b.array[j].col = this->array[i].row;
                b.array[j].val = this->array[i].val;

                buf[array[i].col]++;
            }

            delete[] buf;
        }

        return std::move(b);
    }

    // O(nzz)
    SparseCSR SparseCSR::Add(const SparseCSR &b)
    {
        // std::assert(this->rows == b.rows && this->cols == b.cols);

        SparseCSR c(this->rows, this->cols, this->nnz + b.nnz);
        int i = 0;
        int j = 0;
        int p = 0;

        while (i < this->nnz && j < b.nnz)
        {
            ElementCOO aCOO = this->array[i];
            ElementCOO bCOO = b.array[j];

            int aIndex = aCOO.row * cols + aCOO.col;
            int bIndex = bCOO.row * cols + bCOO.col;

            if (aIndex < bIndex)
            {
                c.array[p] = aCOO;
                p++;
                i++;
            }
            else if (aIndex > bIndex)
            {
                c.array[p] = bCOO;
                p++;
                j++;
            }
            else
            {
                c.array[p] = aCOO;
                c.array[p].val += bCOO.val;
                p++;
                i++;
                j++;
            }
        }

        while (i < this->nnz)
        {
            c.array[p] = this->array[i];
            p++;
            i++;
        }

        while (j < b.nnz)
        {
            c.array[p] = b.array[j];
            p++;
            j++;
        }

        // resize non-zero element
        if (p < this->nnz + b.nnz)
        {
            ElementCOO *_array = new ElementCOO[p];
            std::copy_n(c.array, p, _array);
            delete[] c.array;
            c.array = _array;

            c.nnz = p;
        }

        c.__buildRowStartFromArray();

        return std::move(c);
    }

    SparseCSR SparseCSR::Sub(const SparseCSR &b)
    {
        // std::assert(this->rows == b.rows && this->cols == b.cols);

        // TODO

        return SparseCSR();
    }

    // O(nzz)
    SparseCSR SparseCSR::Mul(const SparseCSR &b)
    {
        // std::assert(this->cols == b.rows);
        SparseCSR c(this->rows, b.cols, 0);

        std::vector<ElementCOO> _array;

        std::vector<float> tmpRowOfC(c.cols); // save a row of matrix C

        int cur = 0;

        while (cur < this->nnz)
        {
            std::fill(tmpRowOfC.begin(), tmpRowOfC.end(), 0.0);

            // calculate the rowA-th row of C

            int rowA = this->array[cur].row;
            while (cur < this->nnz && this->array[cur].row == rowA)
            {
                int colA = this->array[cur].col;
                for (int i = b.rowStart[colA]; i < b.rowStart[colA + 1]; i++)
                {
                    int colB = b.array[i].col;
                    tmpRowOfC[colB] += this->array[cur].val * b.array[i].val; // +=
                }
                cur++;
            }

            for (int colC = 0; colC < c.cols; colC++)
            {
                if (tmpRowOfC[colC] != 0.0)
                {
                    _array.push_back(ElementCOO{rowA, colC, tmpRowOfC[colC]});
                }
            }
        }

        // foolish code !!!
        c.nnz = _array.size();
        c.array = new ElementCOO[c.nnz];
        std::copy_n(_array.data(), c.nnz, c.array);

        c.__buildRowStartFromArray();

        return std::move(c);
    }

} // namespace sparse