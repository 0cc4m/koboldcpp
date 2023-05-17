#pragma once

#include "ggml.h"

#define CL_TARGET_OPENCL_VERSION 110
#include <CL/cl.h>

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_cl_init(void);

bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

void * ggml_cl_host_malloc(size_t size, cl_mem* mem);
void   ggml_cl_host_free(void * ptr, cl_mem* mem);

void ggml_cl_transform_tensor(struct ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
