#define HASHLEN 32

#define CL_SILENCE_DEPRECATION 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <unistd.h>
#include "../mochimo.h"
#include "../proto.h"
#include "../crypto/sha256.h"
#include "../config.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

typedef struct __trigg_cuda_ctx {
    cl_uchar curr_seed[16], next_seed[16];
    cl_char cp[256], *next_cp;
    cl_mem d_foundObj, d_seedObj, c_blockNumber8, c_midstate256, c_input32;
    cl_int *found;
    cl_uchar *seed;
    cl_uint *midstate, *input;
    cl_command_queue cmd_queue;
    size_t local_group_size;
} TriggCTX;

/* Max 64 GPUs Supported */
#define MAXDEV 64
TriggCTX ctx[MAXDEV];
cl_device_id devices[MAXDEV];
size_t threads = 600047615;
cl_char *nullcp = NULL;
byte diff;
byte *bnum;
cl_uint nGPU = -1;
cl_context context = NULL;
cl_kernel kernel = NULL;

void setup_kernel()
{
    cl_uint num_platforms;
    cl_platform_id platform_id = NULL;
    cl_int ret;
    FILE *fp;
    char fileName[] = "trigg.cl";
    char *source_str;
    size_t source_size;
    cl_program program = NULL;

    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, MAXDEV, devices, &nGPU);
//    printf("%d gpu devices found\n", nGPU);

    /* Create OpenCL context */
    context = clCreateContext(NULL, nGPU, devices, NULL, NULL, &ret);

    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                        (const size_t *)&source_size, &ret);
    /* Build Kernel Program */
    ret = clBuildProgram(program, nGPU, devices, NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        unsigned long len = 0;
        ret = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char *buffer = calloc(len, sizeof(char));
        ret = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("\n\nBuildlog:   %s\n\n",buffer);
    }

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "trigg", &ret);
    if (ret != CL_SUCCESS) {
        printf("Error kernel\n");
    }
    free(source_str);
}

int trigg_init_cl(byte difficulty, byte *blockNumber) {

    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_int ret;
    cl_int fill_value = 0;

    /* Obtain and check system GPU count */
    if(nGPU < 1 || nGPU > 64) return nGPU;

    diff = difficulty;
    bnum = malloc(8);
    memcpy(bnum, blockNumber, 8);

    int i = 0;
    for (; i < nGPU; i++) {

        ctx[i].cmd_queue = clCreateCommandQueue(context, devices[i], 0, &ret);

        ctx[i].d_foundObj = clCreateBuffer(context, CL_MEM_READ_WRITE, 4, NULL, &ret);
        ctx[i].d_seedObj = clCreateBuffer(context, CL_MEM_READ_WRITE, 16, NULL, &ret);
        ctx[i].c_blockNumber8 = clCreateBuffer(context, CL_MEM_READ_ONLY, 8, NULL, &ret);
//        ret = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void *)&ctx[i].local_group_size, NULL);

        ret = clEnqueueWriteBuffer(ctx[i].cmd_queue, ctx[i].c_blockNumber8, CL_FALSE, 0,
                                  8, bnum, 0, NULL, NULL);

        ctx[i].found = malloc(4);
        ctx[i].seed = malloc(16);
        ctx[i].midstate = malloc(32);
        ctx[i].input = malloc(32);

        /* Set remaining device memory */
        ret = clEnqueueFillBuffer(ctx[i].cmd_queue, ctx[i].d_foundObj, &fill_value, 1, 0, sizeof(cl_uint), 0, NULL, NULL);
        ret = clEnqueueFillBuffer(ctx[i].cmd_queue, ctx[i].d_seedObj, &fill_value, 1, 0, 16, 0, NULL, NULL);

        /* Setup variables for "first round" */
        *ctx[i].found = 0;
        ctx[i].next_cp = nullcp;
    }

    return nGPU;
}

/*
void trigg_free_cl() {
    int i = 0;
    for(; i < nGPU; i++) {

        free(ctx[i].found);
        free(ctx[i].seed);
        free(ctx[i].midstate);
        free(ctx[i].input);
        free(bnum);

        clReleaseMemObject(ctx[i].d_foundObj);
        clReleaseMemObject(ctx[i].d_seedObj);
        clReleaseMemObject(ctx[i].c_blockNumber8);
        clReleaseMemObject(ctx[i].c_midstate256);
        clReleaseMemObject(ctx[i].c_input32);
        clReleaseCommandQueue(ctx[i].cmd_queue);
    }
}*/

extern byte Tchain[32 + 256 + 16 + 8];
extern byte *trigg_gen(byte *in);
extern char *trigg_expand(byte *in, int diff);
extern char *trigg_check(byte *in, byte d, byte *bnum);

char *gen_haiku(byte *mroot, unsigned long long *nHaiku)
{
    int ret;
    int i;
    for (i = 0; i < nGPU; i++) {
        if(ctx[i].next_cp == nullcp) {
            trigg_gen(ctx[i].next_seed);
            ctx[i].next_cp = (cl_char *)trigg_expand(ctx[i].next_seed, diff);

            memcpy(Tchain, mroot, 32);

            SHA256_CTX sha256;
            sha256_init(&sha256);
            sha256_update(&sha256, Tchain, 256);
            memcpy(ctx[i].midstate, sha256.state, 32);
            memcpy(ctx[i].input, Tchain + 256, 32);
        }

        if(*ctx[i].found<0) continue;

        if(*ctx[i].found<1) {
            ctx[i].c_midstate256 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32, NULL, &ret);
            ctx[i].c_input32 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32, NULL, &ret);

            ret = clEnqueueWriteBuffer(ctx[i].cmd_queue, ctx[i].c_midstate256, CL_FALSE, 0,
                                       32, ctx[i].midstate, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(ctx[i].cmd_queue, ctx[i].c_input32, CL_FALSE, 0,
                                       32, ctx[i].input, 0, NULL, NULL);

            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ctx[i].d_foundObj);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &ctx[i].d_seedObj);
            ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &ctx[i].c_input32);
            ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &ctx[i].c_midstate256);
            ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &ctx[i].c_blockNumber8);
            cl_uint df = diff;
            ret = clSetKernelArg(kernel, 5, sizeof(cl_uint), &df);

//            size_t maxWorkGroupSize;
//            clGetKernelWorkGroupInfo(kernel, devices[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL );

            ret = clEnqueueNDRangeKernel(ctx[i].cmd_queue, kernel, 1, 0, &threads, NULL, 0, NULL, NULL);
            if (ret != CL_SUCCESS) {
                printf("Err %d\n", ret);
            }
            ret = clEnqueueReadBuffer(ctx[i].cmd_queue, ctx[i].d_foundObj, CL_FALSE, 0, 4, ctx[i].found, 0, NULL, NULL);

            *nHaiku += threads;
            *ctx[i].found = -1;

            memcpy(ctx[i].curr_seed,ctx[i].next_seed,16);
            strcpy((char *)ctx[i].cp, (char *)ctx[i].next_cp);
            ctx[i].next_cp = nullcp;
            continue;
        }

        ret = clEnqueueReadBuffer(ctx[i].cmd_queue, ctx[i].d_seedObj, CL_TRUE, 0, 16, ctx[i].seed, 0, NULL, NULL);
        memcpy(mroot + 32, ctx[i].curr_seed, 16);
        memcpy(mroot + 32 + 16, ctx[i].seed, 16);
        return (char *)ctx[i].cp;
    }
    return NULL;
}
