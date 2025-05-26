#include "opencl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define VECTOR_SIZE 3

// Simple OpenCL kernel for vector addition
const char *kernelSource =
"__kernel void vector_add(__global const int *A, \n"
"                         __global const int *B, \n"
"                         __global int *C) {    \n"
"    int i = get_global_id(0);                  \n"
"    C[i] = A[i] + B[i];                        \n"
"}                                              \n";

int perform_simple_opencl_operation(void) {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem mem_A = NULL;
    cl_mem mem_B = NULL;
    cl_mem mem_C = NULL;
    cl_int ret;

    int A[VECTOR_SIZE] = {1, 2, 3};
    int B[VECTOR_SIZE] = {4, 5, 6};
    int C[VECTOR_SIZE];

    // Get platform and device information
    ret = clGetPlatformIDs(1, &platform_id, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform ID: %d\n", ret);
        return 1;
    }

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to get CPU device ID: %d\n", ret);
        // Fallback to GPU if CPU not found or no specific preference
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (ret != CL_SUCCESS) {
            fprintf(stderr, "Failed to get GPU device ID: %d\n", ret);
             // Fallback to default device if GPU also not found
            ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
            if (ret != CL_SUCCESS) {
                fprintf(stderr, "Failed to get any device ID: %d\n", ret);
                return 1;
            }
        }
    }
    
    char deviceName[128];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("Using OpenCL device: %s\n", deviceName);


    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context: %d\n", ret);
        return 1;
    }

    // Create command queue
#ifdef CL_VERSION_2_0
    command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
#else
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
#endif
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue: %d\n", ret);
        clReleaseContext(context);
        return 1;
    }

    // Create memory buffers
    mem_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           VECTOR_SIZE * sizeof(int), A, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer A: %d\n", ret);
        goto cleanup_queue_context;
    }
    mem_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           VECTOR_SIZE * sizeof(int), B, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer B: %d\n", ret);
        goto cleanup_memA_queue_context;
    }
    mem_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           VECTOR_SIZE * sizeof(int), NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer C: %d\n", ret);
        goto cleanup_memB_memA_queue_context;
    }

    // Create program from kernel source
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program: %d\n", ret);
        goto cleanup_memC_memB_memA_queue_context;
    }

    // Build program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to build program: %d\n", ret);
        // Print build log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        goto cleanup_program_memC_memB_memA_queue_context;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "vector_add", &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel: %d\n", ret);
        goto cleanup_program_memC_memB_memA_queue_context;
    }

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_A);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arg 0: %d\n", ret);
        goto cleanup_kernel_program_memC_memB_memA_queue_context;
    }
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_B);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arg 1: %d\n", ret);
        goto cleanup_kernel_program_memC_memB_memA_queue_context;
    }
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_C);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arg 2: %d\n", ret);
        goto cleanup_kernel_program_memC_memB_memA_queue_context;
    }

    // Execute OpenCL kernel
    size_t global_work_size = VECTOR_SIZE;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue kernel: %d\n", ret);
        goto cleanup_kernel_program_memC_memB_memA_queue_context;
    }

    // Read results from memory buffer C
    ret = clEnqueueReadBuffer(command_queue, mem_C, CL_TRUE, 0, VECTOR_SIZE * sizeof(int), C, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to read buffer C: %d\n", ret);
        goto cleanup_kernel_program_memC_memB_memA_queue_context;
    }

    // Print results
    printf("OpenCL Result: C = [");
    for (int i = 0; i < VECTOR_SIZE; i++) {
        printf("%d", C[i]);
        if (i < VECTOR_SIZE - 1) {
            printf(",");
        }
    }
    printf("]\n");

    // Cleanup
cleanup_kernel_program_memC_memB_memA_queue_context:
    if(kernel) clReleaseKernel(kernel);
cleanup_program_memC_memB_memA_queue_context:
    if(program) clReleaseProgram(program);
cleanup_memC_memB_memA_queue_context:
    if(mem_C) clReleaseMemObject(mem_C);
cleanup_memB_memA_queue_context:
    if(mem_B) clReleaseMemObject(mem_B);
cleanup_memA_queue_context:
    if(mem_A) clReleaseMemObject(mem_A);
cleanup_queue_context:
    if(command_queue) clReleaseCommandQueue(command_queue);
    if(context) clReleaseContext(context);

    return (ret == CL_SUCCESS) ? 0 : 1;
}
