# HPC Lab1

## 1. gemm

### 1.1 implementation

- trival implementation (speedup 1x)
- soft optimized implementation (speedup 30x, **required: The dimension of the matrix is the multiple of 32.**)
- algorithm optimized implementation: Strassen (speedup 60x, **required: The dimension of the matrix is the multiple of 32.**)

### 1.2 reference

- https://software.intel.com/sites/default/files/m/c/d/5/3/d/24469-Strassen_akki.pdf 

- https://software.intel.com/content/www/us/en/develop/articles/performance-of-classic-matrix-multiplication-algorithm-on-intel-xeon-phi-processor-system.html

---------------

## 2. sparse matrix

### 2.1 implementation

- CSR format store
- matrix transpose
- matrix addition
- matrix multiplication

### 2.2 reference

- 《数据结构（C++版）》第二版，殷人昆 主编，清华大学出版社，章节 4.3


----------