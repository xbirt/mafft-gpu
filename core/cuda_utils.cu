#include "cuda_utils.h"
#include <cstdio>   // For printf
#include <cstdlib>  // For exit, EXIT_FAILURE

// GPU Memory Management Definitions
void* cuda_allocate_memory(size_t size) {
    void* devPtr;
    CUDA_CHECK_ERROR(cudaMalloc(&devPtr, size));
    return devPtr;
}

void cuda_free_memory(void* devPtr) {
    if (devPtr) {
        CUDA_CHECK_ERROR(cudaFree(devPtr));
    }
}

void cuda_memcpy_to_device(void* dst, const void* src, size_t count) {
    CUDA_CHECK_ERROR(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void cuda_memcpy_to_host(void* dst, const void* src, size_t count) {
    CUDA_CHECK_ERROR(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

// Device Management Definitions
int get_gpu_device_count() {
    int count;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&count));
    return count;
}

void select_gpu(int deviceId) {
    int device_count = get_gpu_device_count();
    if (deviceId < 0 || deviceId >= device_count) {
        fprintf(stderr, "Error: Invalid GPU device ID %d. Found %d devices.\n", deviceId, device_count);
        fprintf(stderr, "Using device 0 instead.\n");
        deviceId = 0;
    }
    if (device_count > 0) {
      CUDA_CHECK_ERROR(cudaSetDevice(deviceId));
      printf("Selected GPU device: %d\n", deviceId);
    } else {
      fprintf(stderr, "Error: No CUDA-capable devices found.\n");
      exit(EXIT_FAILURE);
    }
}

void get_gpu_properties() {
    int deviceId;
    CUDA_CHECK_ERROR(cudaGetDevice(&deviceId)); // Get current device ID

    cudaDeviceProp properties;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&properties, deviceId));

    printf("--- GPU Device %d: %s ---\n", deviceId, properties.name);
    printf("  Compute Capability: %d.%d\n", properties.major, properties.minor);
    printf("  Total Global Memory: %.2f GB\n", properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Memory per Block: %.2f KB\n", properties.sharedMemPerBlock / 1024.0);
    printf("  Registers per Block: %d\n", properties.regsPerBlock);
    printf("  Warp Size: %d\n", properties.warpSize);
    printf("  Max Threads per Block: %d\n", properties.maxThreadsPerBlock);
    printf("  Max Threads Dimensions: (%d, %d, %d)\n",
           properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
    printf("  Max Grid Dimensions: (%d, %d, %d)\n",
           properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);
    printf("------------------------------------\n");
}

// Add a simple test function that can be called from C code to verify linking
extern "C" void cuda_utils_test_function() {
    printf("CUDA Utils test function executed successfully!\n");
    int dev_count = get_gpu_device_count();
    if (dev_count > 0) {
        select_gpu(0); // Select GPU 0
        get_gpu_properties(); // Print its properties
    } else {
        printf("No CUDA devices found.\n");
    }
}
