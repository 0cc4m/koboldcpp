#include "ggml-opencl.h"

#include <clblast_half.h>

#include <stdio.h>
#include <string.h>

#define MULTILINE_QUOTE(...) #__VA_ARGS__
const char * clblast_dequant = MULTILINE_QUOTE(

struct block_q4_0
{
    float d;
    uchar qs[16];
};

__kernel void dequantize_row_q4_0(__global struct block_q4_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;

    const uchar vi = blocks[i].qs[l];

    const uint index = i*32 + l*2;
    result[index + 0] = ((vi & 0xf) - 8)*d;
    result[index + 1] = ((vi >> 4) - 8)*d;
}

struct block_q4_1
{
    float d;
    float m;
    uchar qs[16];
};

__kernel void dequantize_row_q4_1(__global struct block_q4_1* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;
    const float m = blocks[i].m;

    const uchar vi = blocks[i].qs[l];

    const uint index = i*32 + l*2;
    result[index + 0] = (vi & 0xf) * d + m;
    result[index + 1] = (vi >> 4) * d + m;
}

struct block_q4_2
{
    ushort d;
    uchar qs[8];
};

__kernel void dequantize_row_q4_2(__global struct block_q4_2* blocks, __global float* result) {
    const uint i = get_global_id(0) / 16;
    const uint l = get_local_id(0);

    const float d = vload_half(0, (__global half*) &blocks[i].d);

    const uchar vi = blocks[i].qs[l];

    const uint index = i*16 + l*2;
    result[index + 0] = ((vi & 0xf) - 8)*d;
    result[index + 1] = ((vi >> 4) - 8)*d;
}


struct block_q5_0
{
    float d;
    uint qh;
    uchar qs[16];
};

__kernel void dequantize_row_q5_0(__global struct block_q5_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;

    const uchar vi = blocks[i].qs[l];

    const uint l2 = l * 2;

    const uchar vh0 = ((blocks[i].qh & (1 << (l2 + 0))) >> (l2 + 0)) << 4;
    const uchar vh1 = ((blocks[i].qh & (1 << (l2 + 1))) >> (l2 + 1)) << 4;

    const uint index = i*32 + l2;
    result[index + 0] = (((vi & 0xf) | vh0) - 16)*d;
    result[index + 1] = (((vi >>  4) | vh1) - 16)*d;
}

struct block_q5_1
{
    ushort d;
    ushort m;
    uint qh;
    uchar qs[16];
};

__kernel void dequantize_row_q5_1(__global struct block_q5_1* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = vload_half(0, (__global half*) &blocks[i].d);
    const float m = vload_half(0, (__global half*) &blocks[i].m);

    const uchar vi = blocks[i].qs[l];

    const uint l2 = l * 2;

    const uchar vh0 = ((blocks[i].qh & (1 << (l2 + 0))) >> (l2 + 0)) << 4;
    const uchar vh1 = ((blocks[i].qh & (1 << (l2 + 1))) >> (l2 + 1)) << 4;

    const uint index = i*32 + l2;
    result[index + 0] = ((vi & 0xf) | vh0)*d + m;
    result[index + 1] = ((vi >>  4) | vh1)*d + m;
}

struct block_q8_0
{
    float d;
    char qs[32];
};

__kernel void dequantize_row_q8_0(__global struct block_q8_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    result[i*32 + l] = blocks[i].qs[l] * blocks[i].d;
}

);

#define CL_CHECK(err, name)                                                                     \
    do {                                                                                        \
        cl_int err_ = (err);                                                                    \
        if (err_ != CL_SUCCESS) {                                                               \
            fprintf(stderr, "OpenCL %s error %d at %s:%d\n", name, err_, __FILE__, __LINE__);   \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

#define QK5_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;


typedef struct {
    float d;                // delta
    uint32_t qh;            // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} cl_block_q5_0;

static cl_platform_id platform;
static cl_device_id device;
static bool fp16_support;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel_q4_0, kernel_q4_1, kernel_q4_2, kernel_q5_0, kernel_q5_1, kernel_q8_0;
static cl_mem cl_buffer_a, cl_buffer_qa, cl_buffer_b, cl_buffer_c;
static size_t cl_size_a = 0, cl_size_qa = 0, cl_size_b = 0, cl_size_c = 0;

static cl_kernel kernel;
static size_t global, local, size_qa;
static bool dequant;
static bool fp16;
static cl_block_q5_0* cl_host_a;

static cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer) {
    cl_program p;
    char *program_log;
    size_t program_size, log_size;
    int err;

    program_size = strlen(program_buffer);

    p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        fprintf(stderr, "OpenCL error creating program");
        exit(1);
    }

    err = clBuildProgram(p, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {

        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return p;
}

void ggml_cl_init(void) {
    cl_int err = 0;
    char * GGML_CLBLAST_PLATFORM = getenv("GGML_CLBLAST_PLATFORM");
    char * GGML_CLBLAST_DEVICE = getenv("GGML_CLBLAST_DEVICE");
    int plat_num = (GGML_CLBLAST_PLATFORM == NULL ? 0 : atoi(GGML_CLBLAST_PLATFORM));
    int dev_num = (GGML_CLBLAST_DEVICE == NULL ? 0 : atoi(GGML_CLBLAST_DEVICE));
    printf("\nInitializing CLBlast (First Run)...");
    printf("\nAttempting to use: Platform=%d, Device=%d (If invalid, program will crash)\n",plat_num,dev_num);
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    platform = platforms[plat_num];
    char platform_buffer[1024];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_buffer), &platform_buffer, NULL);
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    cl_device_id* devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    device = devices[dev_num];
    char device_buffer[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_buffer), &device_buffer, NULL);
    size_t ext_str_size;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_str_size);
    char* ext_buffer = (char*) malloc(sizeof(char) * ext_str_size);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_str_size, ext_buffer, NULL);
    // Check if ext_buffer contains cl_khr_fp16
    for (size_t i = 0; i < ext_str_size - 12; i++) {
        if (memcmp(ext_buffer + i, "cl_khr_fp16", 11) == 0) {
            fp16_support = true;
            break;
        }
    }
    free(ext_buffer);
    printf("Using Platform: %s Device: %s FP16: %d\n", platform_buffer, device_buffer, fp16_support);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");
    queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    free(platforms);
    free(devices);

    program = build_program_from_source(context, device, clblast_dequant);

    // Prepare dequantize kernels
    kernel_q4_0 = clCreateKernel(program, "dequantize_row_q4_0", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q4_1 = clCreateKernel(program, "dequantize_row_q4_1", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q4_2 = clCreateKernel(program, "dequantize_row_q4_2", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q5_0 = clCreateKernel(program, "dequantize_row_q5_0", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q5_1 = clCreateKernel(program, "dequantize_row_q5_1", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q8_0 = clCreateKernel(program, "dequantize_row_q8_0", &err);
    CL_CHECK(err, "clCreateKernel");
}

bool ggml_cl_fp16(void) {
    return fp16_support;
}

static void ggml_cl_malloc(size_t req_size, size_t* cur_size, cl_mem_flags flags, cl_mem* buf) {
    if (req_size <= *cur_size) {
        return;
    }

    // Reallocate buffer with enough space
    if (*cur_size > 0) {
        clReleaseMemObject(*buf);
    }
    cl_int err;
    *buf = clCreateBuffer(context, flags, req_size, NULL, &err);
    *cur_size = req_size;
    CL_CHECK(err, "clCreateBuffer");
}

static cl_int ggml_cl_h2d_tensor_2d(cl_command_queue queue, cl_mem dst, const struct ggml_tensor * src, uint64_t i3, uint64_t i2, cl_event* ev) {
    cl_int err;
    const uint64_t ne0 = src->ne[0];
    const uint64_t ne1 = src->ne[1];
    const uint64_t nb0 = src->nb[0];
    const uint64_t nb1 = src->nb[1];
    const uint64_t nb2 = src->nb[2];
    const uint64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const size_t ts = ggml_type_size(type);
    const size_t bs = ggml_blck_size(type);

    const void * x = (const void *) ((const char *) src->data + i2*nb2 + i3*nb3);
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        err = clEnqueueWriteBuffer(queue, dst, CL_FALSE, 0, ne1*nb1, x, 0, NULL, ev);
        return err;
    } else if (nb0 == ts) {
        const size_t origin[3] = { 0, 0, 0 };
        const size_t region[3] = { ts*ne0/bs, ne1, 1 };
        err = clEnqueueWriteBufferRect(queue, dst, CL_FALSE, origin, origin, region, ts*ne0/bs, 0, nb1, 0, x, 0, NULL, ev);
        return err;
    } else {
        for (uint64_t i1 = 0; i1 < ne1; i1++) {
            // pretend the row is a matrix with cols=1
            const size_t buffer_origin[3] = { 0, i1, 0 };
            const size_t host_origin[3] = { 0, 0, 0 };
            const size_t region[3] = { ts/bs, ne0, 1 };
            err = clEnqueueWriteBufferRect(queue, dst, CL_FALSE, buffer_origin, host_origin, region, 0, 0, nb0, 0, ((const char *)x) + i1*nb0, 0, NULL, ev);
            if (err != CL_SUCCESS) {
                break;
            }
        }
        return err;
    }
}

void ggml_cl_blas_setup(size_t size_a, size_t size_b, size_t size_c, int btype) {
    global = size_a;

    fp16 = false;

    switch (btype) {
    case GGML_TYPE_F16:
        dequant = false;
        if (fp16_support) {
            fp16 = true;
        }
        break;
    case GGML_TYPE_F32:
        dequant = false;
        break;
    case GGML_TYPE_Q4_0:
        dequant = true;
        kernel = kernel_q4_0;
        local = 16;
        size_qa = global * (sizeof(float) + local) / 32;
        break;
    case GGML_TYPE_Q4_1:
        dequant = true;
        kernel = kernel_q4_1;
        local = 16;
        size_qa = global * (sizeof(float) * 2 + local) / 32;
        break;
    case GGML_TYPE_Q4_2:
        dequant = true;
        kernel = kernel_q4_2;
        local = 8;
        size_qa = global * (sizeof(ggml_fp16_t) + local) / 16;
        break;
    case GGML_TYPE_Q5_0:
        dequant = true;
        kernel = kernel_q5_0;
        local = 16;
        cl_host_a = (cl_block_q5_0*) malloc(sizeof(cl_block_q5_0) * global / 32);
        size_qa = global * (sizeof(float) + sizeof(uint32_t) + local) / 32;
        break;
    case GGML_TYPE_Q5_1:
        dequant = true;
        kernel = kernel_q5_1;
        local = 16;
        size_qa = global * (sizeof(ggml_fp16_t) * 2 + sizeof(uint32_t) + local) / 32;
        break;
    case GGML_TYPE_Q8_0:
        dequant = true;
        kernel = kernel_q8_0;
        local = 32;
        size_qa = global * (sizeof(float) + local) / 32;
        break;
    default:
        fprintf(stderr, "Error: Unsupported OpenCL btype %d\n", btype);
        abort();
    }

    // Prepare buffers
    if (dequant) {
        ggml_cl_malloc(size_qa, &cl_size_qa, CL_MEM_READ_ONLY, &cl_buffer_qa);
    }
    ggml_cl_malloc((fp16 ? sizeof(ggml_fp16_t) : sizeof(float)) * size_a, &cl_size_a, CL_MEM_READ_WRITE, &cl_buffer_a);
    ggml_cl_malloc((fp16 ? sizeof(ggml_fp16_t) : sizeof(float)) * size_b, &cl_size_b, CL_MEM_READ_ONLY, &cl_buffer_b);
    ggml_cl_malloc((fp16 ? sizeof(ggml_fp16_t) : sizeof(float)) * size_c, &cl_size_c, CL_MEM_WRITE_ONLY, &cl_buffer_c);
}

void ggml_cl_blas_init_buffer(const void* host, size_t size, bool buffer_a) {
    cl_int err;

    err = clEnqueueWriteBuffer(queue, buffer_a ? cl_buffer_a : cl_buffer_b, CL_FALSE, 0, (fp16 ? sizeof(ggml_fp16_t) : sizeof(float)) * size, host, 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueWriteBuffer a");
}
void ggml_cl_blas_init_tensor(const struct ggml_tensor* host, uint64_t i3, uint64_t i2, bool buffer_a) {
    cl_int err;

    err = ggml_cl_h2d_tensor_2d(queue, buffer_a ? cl_buffer_a : cl_buffer_b, host, i3, i2, NULL);
    CL_CHECK(err, "ggml_cl_h2d_tensor_2d b");
}

void ggml_cl_blas_init_dequant(const struct ggml_tensor* host_a, const struct ggml_tensor* host_b, uint64_t i3, uint64_t i2, int btype) {
    cl_int err;
    cl_event ev_qa;

    if (dequant) {
        if (btype == GGML_TYPE_Q5_0) {
            // For some reason OpenCL seems to be incapable of working with structs of size 22.
            // 20 and 24 bytes are fine. Workaround to do the fp16 to fp32 step on CPU...
            // TODO Find the reason, fix and remove workaround.
            const block_q5_0* a = (const block_q5_0*) host_a;
            for (size_t i = 0; i < global / 32; i++) {
                cl_host_a[i].d = ggml_fp16_to_fp32(a[i].d);
                memcpy(&cl_host_a[i].qh, a[i].qh, sizeof(uint32_t) + QK5_0 / 2);
            }
            host_a = (const float*) cl_host_a;
        }
        err = ggml_cl_h2d_tensor_2d(queue, cl_buffer_qa, host_a, i3, i2, &ev_qa);
        CL_CHECK(err, "ggml_cl_h2d_tensor_2d qa");
    } else {
        err = ggml_cl_h2d_tensor_2d(queue, cl_buffer_a, host_a, i3, i2, NULL);
        CL_CHECK(err, "ggml_cl_h2d_tensor_2d a");
    }

    err = ggml_cl_h2d_tensor_2d(queue, cl_buffer_b, host_b, i3, i2, NULL);
    CL_CHECK(err, "ggml_cl_h2d_tensor_2d b");
    if (dequant) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qa);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_a);
        CL_CHECK(err, "clSetKernelArg");
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 1, &ev_qa, NULL);
        CL_CHECK(err, "clEnqueueNDRangeKernel");
        clReleaseEvent(ev_qa);
    }
}

void ggml_cl_blas_run(
        const enum ggml_blas_order order, const enum ggml_blas_op trans_a, const enum ggml_blas_op trans_b,
        const int m, const int n, const int k,
        const float alpha, const int lda,
                           const int ldb,
        const float beta, float *host_c, const int ldc,
        const int btype) {
    cl_int err = 0;
    cl_event ev_gemm, ev_c;

    clFinish(queue);

    if (btype == GGML_TYPE_F32) {
        CLBlastStatusCode status = CLBlastSgemm((CLBlastLayout)order,
                                                (CLBlastTranspose)trans_a, (CLBlastTranspose)trans_b,
                                                m, n, k,
                                                alpha,
                                                cl_buffer_a, 0, lda,
                                                cl_buffer_b, 0, ldb,
                                                beta,
                                                cl_buffer_c, 0, ldc,
                                                &queue, &ev_gemm);

        if (status != CLBlastSuccess) {
            fprintf(stderr, "Error: CLBlast SGEMM %d\n", status);
            abort();
        }

        clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, sizeof(float) * m * n, host_c, 1, &ev_gemm, &ev_c);
    } else if (btype == GGML_TYPE_F16) {
        CLBlastStatusCode status = CLBlastHgemm((CLBlastLayout)order,
                                                (CLBlastTranspose)trans_a, (CLBlastTranspose)trans_b,
                                                m, n, k,
                                                FloatToHalf(alpha),
                                                cl_buffer_a, 0, lda,
                                                cl_buffer_b, 0, ldb,
                                                FloatToHalf(beta),
                                                cl_buffer_c, 0, ldc,
                                                &queue, &ev_gemm);

        if (status != CLBlastSuccess) {
            fprintf(stderr, "Error: CLBlast HGEMM %d\n", status);
            abort();
        }

        cl_half* buffer = (cl_half*) malloc(sizeof(cl_half) * m * n);

        clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, sizeof(cl_half) * m * n, buffer, 1, &ev_gemm, &ev_c);

        for (size_t i = 0; i < (size_t)(m * n); i++) {
            host_c[i] = HalfToFloat(buffer[i]);
        }

        free(buffer);
    } else {
        fprintf(stderr, "Error: Unsupported CLBlast GEMM type %d\n", btype);
        abort();
    }

    // Wait for completion
    clWaitForEvents(1, &ev_c);
    clReleaseEvent(ev_gemm);
    clReleaseEvent(ev_c);
}

void ggml_cl_blas_cleanup(int btype) {
    if (btype == GGML_TYPE_Q5_0) {
        free((void*) cl_host_a);
    }
}
