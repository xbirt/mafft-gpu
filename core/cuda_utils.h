#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h> // For cudaError_t and other CUDA types

#ifdef __cplusplus
#include <cstdio>       // For fprintf, stderr
#include <cstdlib>      // For exit, EXIT_FAILURE
#else
#include <stdio.h>      // For C: fprintf, stderr
#include <stdlib.h>     // For C: exit, EXIT_FAILURE
#endif

// CUDA Error Checking Macro
// This macro will wrap CUDA API calls and print an error message if they fail.
#define CUDA_CHECK_ERROR(err)     do {         cudaError_t err_ = (err);         if (err_ != cudaSuccess) {             fprintf(stderr, "CUDA error in %s at line %d: %s\n",                     __FILE__, __LINE__, cudaGetErrorString(err_));             exit(EXIT_FAILURE);         }     } while (0)


#ifdef __cplusplus
extern "C" {
#endif

// GPU Memory Management Prototypes
void* cuda_allocate_memory(size_t size);
void cuda_free_memory(void* devPtr);
void cuda_memcpy_to_device(void* dst, const void* src, size_t count);
void cuda_memcpy_to_host(void* dst, const void* src, size_t count);

// Device Management Prototypes
void select_gpu(int deviceId);
void get_gpu_properties();
int get_gpu_device_count();
void cuda_utils_test_function(); // Declaration for the test function

#ifdef __cplusplus
}
#endif

#endif // CUDA_UTILS_H
