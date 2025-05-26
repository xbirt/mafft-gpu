#ifndef OPENCL_ALIGN_H
#define OPENCL_ALIGN_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Function to perform global alignment using OpenCL.
// Similar to G__align11, but tailored for OpenCL.
// Note: Modifying seq1_cpu[0] and seq2_cpu[0] with the alignment result
// will require careful handling of memory allocated by this function vs. by the caller.
// For an initial step, this function might return new char* for aligned sequences,
// or expect sufficiently large buffers. G__align11 modifies its inputs directly.
double G__align11_cl(
    cl_platform_id platform_id, // Allow passing existing platform/device/context/queue
    cl_device_id device_id,     // to avoid re-initialization for multiple calls.
    cl_context context,         // If NULL, this function will set them up.
    cl_command_queue command_queue,
    double **n_dynamicmtx,      // Scoring matrix (nalphabets x nalphabets)
    const char *seq1_str,       // Sequence 1 string
    const char *seq2_str,       // Sequence 2 string
    int headgp,                 // Head gap penalty flag
    int tailgp,                 // Tail gap penalty flag
    float penalty_open,         // Gap open penalty (already includes extend)
    float penalty_extend,       // Gap extend penalty
    char *aligned_seq1_out,     // Buffer to store aligned sequence 1
    char *aligned_seq2_out,     // Buffer to store aligned sequence 2
    int buffer_size             // Size of aligned_seq1_out and aligned_seq2_out
);

#endif // OPENCL_ALIGN_H
