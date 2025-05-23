#ifndef DP_KERNELS_H
#define DP_KERNELS_H

#include "cuda_utils.h" // For CUDA types and utilities
#include "mltaln.h"     // For MAFFT data structures (ensure this is accessible)
                        // May need to adjust include paths or forward declare if direct include is problematic.

#ifdef __cplusplus
extern "C" {
#endif

// Example kernel launcher prototype (will be refined)
// Parameters will depend on the actual data needed by the DP algorithm.
// We'll need to pass pointers to sequences, scoring matrices, gap penalties,
// dimensions, and pointers to GPU memory for results.
// For now, a placeholder:
void launch_dp_kernel(
    char **seq1_gpu, 
    char **seq2_gpu,
    double **scoring_matrix_gpu, 
    int lgth1, 
    int lgth2, 
    double *results_gpu // Placeholder for DP matrix or scores
);

// Add a simple test function to verify compilation and linking
void dp_kernels_test_function();

#ifdef __cplusplus
}
#endif

#endif // DP_KERNELS_H
