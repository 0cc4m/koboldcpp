#pragma once

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast_c.h>

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_cl_init(void);
bool ggml_cl_fp16(void);

enum ggml_blas_order {
    GGML_BLAS_ORDER_ROW_MAJOR = 101,
    GGML_BLAS_ORDER_COLUMN_MAJOR = 102,
};

enum ggml_blas_op {
    GGML_BLAS_OP_N = 111,
    GGML_BLAS_OP_T = 112,
    GGML_BLAS_OP_C = 113,
};

void ggml_cl_blas_setup(size_t size_a, size_t size_b, size_t size_c, int btype);

void ggml_cl_blas_init_buffer(const void* host, size_t size, bool buffer_a);
void ggml_cl_blas_init_tensor(const struct ggml_tensor* host, uint64_t i3, uint64_t i2, bool buffer_a);
void ggml_cl_blas_init_dequant(const struct ggml_tensor* host_a, const struct ggml_tensor* host_b, uint64_t i3, uint64_t i2, int btype);

void ggml_cl_blas_run(
        const enum ggml_blas_order order, const enum ggml_blas_op trans_a, const enum ggml_blas_op trans_b,
        const int m, const int n, const int k,
        const float alpha, const int lda,
                           const int ldb,
        const float beta, float *host_c, const int ldc,
        const int btype);

void ggml_cl_blas_cleanup(int btype);

#ifdef  __cplusplus
}
#endif
