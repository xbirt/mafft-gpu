#include "opencl_align.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h> // For FLT_MAX (or DBL_MAX if using double for NEGATIVE_INFINITY)
#include <math.h>  // For fmaxf, fminf

// mltaln.h or equivalent for amino_n etc.
// G__align11 uses 0x100 for matrix dimensions, so we'll use that for nalphabets_stride.
#define NALPHABETS_STRIDE 0x100 
#define NEGATIVE_INFINITY (-FLT_MAX) // Using -FLT_MAX for a very small float number

#define MAX_SOURCE_SIZE (0x100000) 

// Traceback directions (must match kernel definitions)
#ifndef TRACE_DIAG
#define TRACE_DIAG 0
#endif
#ifndef TRACE_UP
#define TRACE_UP   1 
#endif
#ifndef TRACE_LEFT
#define TRACE_LEFT 2
#endif

// OpenCL kernel string for Gotoh global alignment
const char *dp_kernel_source_str =
"// Traceback directions\n"
"#define TRACE_DIAG 0\n"
"#define TRACE_UP   1 // Gap in sequence 2 (corresponds to Ix)\n"
"#define TRACE_LEFT 2 // Gap in sequence 1 (corresponds to Iy)\n"
"\n"
"// Utility for max of three floats\n"
"inline float max3(float a, float b, float c) {\n"
"    return fmax(a, fmax(b, c));\n"
"}\n"
"\n"
"__kernel void dp_matrix_fill_kernel(\n"
"    __global const int *seq1_int,            // Sequence 1 (integer codes)\n"
"    __global const int *seq2_int,            // Sequence 2 (integer codes)\n"
"    __global const float *scoring_matrix,    // Flattened scoring matrix\n"
"    int seq1_len,                            // Length of sequence 1 (actual, not +1)\n"
"    int seq2_len,                            // Length of sequence 2 (actual, not +1)\n"
"    int nalphabets_stride,                   // Stride for scoring matrix (e.g., 26 for A-Z)\n"
"    float penalty_open_extend,               // Penalty for opening a gap (includes extend)\n"
"    float penalty_extend,                    // Penalty for extending a gap\n"
"    __global float *S_matrix,                // Main score matrix S[i][j]\n"
"    __global float *M_matrix,                // Match/Mismatch score matrix M[i][j]\n"
"    __global float *Ix_matrix,               // Gap in seq2 (X-axis) matrix Ix[i][j]\n"
"    __global float *Iy_matrix,               // Gap in seq1 (Y-axis) matrix Iy[i][j]\n"
"    __global int *traceback_matrix) {        // Traceback direction matrix\n"
"\n"
"    // Kernel computes for cells (i, j) where i from 1..seq1_len, j from 1..seq2_len\n"
"    // Host must have initialized M_matrix[0][0]=0, S_matrix[0][0]=0, Ix/Iy[0][0]=-INF\n"
"    // and row 0 / col 0 for all matrices.\n"
"    int i = get_global_id(0) + 1; // Current row in DP matrix (1-based for seq1)\n"
"    int j = get_global_id(1) + 1; // Current col in DP matrix (1-based for seq2)\n"
"\n"
"    // Ensure we are within the bounds of the sequences to be processed by this kernel instance\n"
"    if (i > seq1_len || j > seq2_len) {\n"
"        return;\n"
"    }\n"
"\n"
"    int s2_stride = seq2_len + 1; // Stride for accessing elements in DP matrices\n"
"\n"
"    // Index for current cell (i,j)\n"
"    int current_idx = i * s2_stride + j;\n"
"    // Indices for neighbor cells\n"
"    int diag_idx  = (i - 1) * s2_stride + (j - 1);\n"
"    int up_idx    = (i - 1) * s2_stride + j;\n"
"    int left_idx  = i * s2_stride + (j - 1);\n"
"\n"
"    // 1. Calculate M_matrix[i][j]\n"
"    //    M[i,j] = score(seq1[i-1], seq2[j-1]) + max(S[i-1,j-1], Ix[i-1,j-1], Iy[i-1,j-1])\n"
"    //    MAFFT G__align11 uses S[i-1,j-1] instead of M[i-1,j-1] for M calculation.\n"
"    //    Correct Gotoh is M[i,j] = s_match(c1,c2) + Max( M(i-1,j-1), Ix(i-1,j-1), Iy(i-1,j-1) )\n"
"    //    Let's stick to the kernel's formula: M_matrix[current_idx] = match_mismatch_score + max3(M_val, Ix_val, Iy_val);\n"
"    int char1_code = seq1_int[i-1];\n"
"    int char2_code = seq2_int[j-1];\n"
"    float match_mismatch_score = scoring_matrix[char1_code * nalphabets_stride + char2_code];\n"
"\n"
"    float prev_m_val  = M_matrix[diag_idx];\n"
"    float prev_ix_val = Ix_matrix[diag_idx];\n"
"    float prev_iy_val = Iy_matrix[diag_idx];\n"
"    M_matrix[current_idx] = match_mismatch_score + max3(prev_m_val, prev_ix_val, prev_iy_val);\n"
"\n"
"    // 2. Calculate Ix_matrix[i][j] (gap in sequence 2 - move right in matrix)\n"
"    //    Ix[i,j] = max( M[i,j-1] + penalty_open_extend, Ix[i,j-1] + penalty_extend )\n"
"    float m_val_for_ix = M_matrix[left_idx];\n"
"    float ix_val_for_ix = Ix_matrix[left_idx];\n"
"    Ix_matrix[current_idx] = fmax(m_val_for_ix + penalty_open_extend,\n"
"                                ix_val_for_ix + penalty_extend);\n"
"\n"
"    // 3. Calculate Iy_matrix[i][j] (gap in sequence 1 - move down in matrix)\n"
"    //    Iy[i,j] = max( M[i-1,j] + penalty_open_extend, Iy[i-1,j] + penalty_extend )\n"
"    float m_val_for_iy = M_matrix[up_idx];\n"
"    float iy_val_for_iy = Iy_matrix[up_idx];\n"
"    Iy_matrix[current_idx] = fmax(m_val_for_iy + penalty_open_extend,\n"
"                                iy_val_for_iy + penalty_extend);\n"
"\n"
"    // 4. Calculate S_matrix[i][j] and determine traceback direction\n"
"    //    S[i,j] = max(M[i,j], Ix[i,j], Iy[i,j])\n"
"    float s_m_val  = M_matrix[current_idx];\n"
"    float s_ix_val = Ix_matrix[current_idx];\n"
"    float s_iy_val = Iy_matrix[current_idx];\n"
"\n"
"    // Simplified tie-breaking: M > Iy > Ix (arbitrary, can be tuned)\n"
"    // This order affects which path is chosen when scores are equal.\n"
"    if (s_m_val >= s_iy_val && s_m_val >= s_ix_val) {\n"
"        S_matrix[current_idx] = s_m_val;\n"
"        traceback_matrix[current_idx] = TRACE_DIAG;\n"
"    } else if (s_iy_val >= s_ix_val) { // Iy is max (or tied with Ix, and M is less)\n"
"        S_matrix[current_idx] = s_iy_val;\n"
"        traceback_matrix[current_idx] = TRACE_UP;\n"
"    } else { // Ix is max\n"
"        S_matrix[current_idx] = s_ix_val;\n"
"        traceback_matrix[current_idx] = TRACE_LEFT;\n"
"    }\n"
"}\n";

double G__align11_cl(
    cl_platform_id platform_id_in, 
    cl_device_id device_id_in,     
    cl_context context_in,         
    cl_command_queue command_queue_in,
    double **n_dynamicmtx,      
    const char *seq1_str,       
    const char *seq2_str,       
    int headgp,                 
    int tailgp,                 
    float penalty_open,         // This is penalty_open_raw + penalty_extend
    float penalty_extend,       
    char *aligned_seq1_out,     
    char *aligned_seq2_out,     
    int buffer_size             
) {
    cl_int ret = CL_SUCCESS; // Initialize ret
    // OpenCL setup variables
    cl_platform_id platform_id = platform_id_in;
    cl_device_id device_id = device_id_in;
    cl_context context = context_in;
    cl_command_queue command_queue = command_queue_in;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem seq1_gpu_buf = NULL, seq2_gpu_buf = NULL, scoring_matrix_gpu_buf = NULL;
    cl_mem S_matrix_gpu_buf = NULL, M_matrix_gpu_buf = NULL, Ix_matrix_gpu_buf = NULL, Iy_matrix_gpu_buf = NULL;
    cl_mem traceback_matrix_gpu_buf = NULL;

    int *seq1_int_host = NULL;
    int *seq2_int_host = NULL;
    float *scoring_matrix_flat_host = NULL;
    float *S_matrix_host = NULL, *M_matrix_host = NULL, *Ix_matrix_host = NULL, *Iy_matrix_host = NULL;
    int *traceback_matrix_host = NULL;

    int seq1_len = strlen(seq1_str);
    int seq2_len = strlen(seq2_str);
    double score = NEGATIVE_INFINITY; // Default score in case of error

    // Determine if we need to setup OpenCL or use provided handles
    int local_cl_setup = (platform_id == NULL);

    if (local_cl_setup) {
        ret = clGetPlatformIDs(1, &platform_id, NULL);
        if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to get platform ID. Error: %d\n", ret); return NEGATIVE_INFINITY; }

        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
        if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to get device ID. Error: %d\n", ret); return NEGATIVE_INFINITY; }

        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create context. Error: %d\n", ret); return NEGATIVE_INFINITY; }

        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
        if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create command queue. Error: %d\n", ret); if(context) clReleaseContext(context); context = NULL; return NEGATIVE_INFINITY; }
    }

    // 1. Sequence Conversion (char to int for kernel)
    seq1_int_host = (int *)malloc(seq1_len * sizeof(int));
    seq2_int_host = (int *)malloc(seq2_len * sizeof(int));
    if (!seq1_int_host || !seq2_int_host) {
        fprintf(stderr, "Failed to allocate memory for host int sequences.\n");
        ret = CL_OUT_OF_HOST_MEMORY; 
        goto cleanup_host_alloc;
    }
    for (int k = 0; k < seq1_len; ++k) seq1_int_host[k] = (unsigned char)seq1_str[k];
    for (int k = 0; k < seq2_len; ++k) seq2_int_host[k] = (unsigned char)seq2_str[k];

    // 2. Scoring Matrix Conversion (double** to float*)
    size_t scoring_matrix_flat_size = NALPHABETS_STRIDE * NALPHABETS_STRIDE * sizeof(float);
    scoring_matrix_flat_host = (float *)malloc(scoring_matrix_flat_size);
    if (!scoring_matrix_flat_host) {
        fprintf(stderr, "Failed to allocate memory for flattened scoring matrix on host.\n");
        ret = CL_OUT_OF_HOST_MEMORY;
        goto cleanup_host_alloc;
    }
    for (int r = 0; r < NALPHABETS_STRIDE; ++r) {
        for (int c = 0; c < NALPHABETS_STRIDE; ++c) {
            scoring_matrix_flat_host[r * NALPHABETS_STRIDE + c] = (float)n_dynamicmtx[r][c];
        }
    }

    // DP matrix dimensions (+1 for boundary conditions)
    size_t dp_rows = seq1_len + 1;
    size_t dp_cols = seq2_len + 1;
    size_t dp_matrix_elems = dp_rows * dp_cols;
    size_t dp_float_matrix_size_bytes = dp_matrix_elems * sizeof(float);
    size_t traceback_matrix_size_bytes = dp_matrix_elems * sizeof(int);

    // 3. OpenCL Buffer Creation
    seq1_gpu_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, seq1_len * sizeof(int), seq1_int_host, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create buffer for seq1. Error: %d\n", ret); goto cleanup_buffers; }
    seq2_gpu_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, seq2_len * sizeof(int), seq2_int_host, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create buffer for seq2. Error: %d\n", ret); goto cleanup_buffers; }
    scoring_matrix_gpu_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scoring_matrix_flat_size, scoring_matrix_flat_host, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create buffer for scoring matrix. Error: %d\n", ret); goto cleanup_buffers; }

    S_matrix_gpu_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dp_float_matrix_size_bytes, NULL, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create buffer for S_matrix. Error: %d\n", ret); goto cleanup_buffers; }
    M_matrix_gpu_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dp_float_matrix_size_bytes, NULL, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create buffer for M_matrix. Error: %d\n", ret); goto cleanup_buffers; }
    Ix_matrix_gpu_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dp_float_matrix_size_bytes, NULL, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create buffer for Ix_matrix. Error: %d\n", ret); goto cleanup_buffers; }
    Iy_matrix_gpu_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dp_float_matrix_size_bytes, NULL, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create buffer for Iy_matrix. Error: %d\n", ret); goto cleanup_buffers; }
    traceback_matrix_gpu_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, traceback_matrix_size_bytes, NULL, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create buffer for traceback_matrix. Error: %d\n", ret); goto cleanup_buffers; }

    // 4. DP Matrix Initialization (0th row/col on CPU, then transfer)
    M_matrix_host = (float *)malloc(dp_float_matrix_size_bytes);
    Ix_matrix_host = (float *)malloc(dp_float_matrix_size_bytes);
    Iy_matrix_host = (float *)malloc(dp_float_matrix_size_bytes);
    S_matrix_host = (float *)malloc(dp_float_matrix_size_bytes);
    if (!M_matrix_host || !Ix_matrix_host || !Iy_matrix_host || !S_matrix_host) {
        fprintf(stderr, "Failed to allocate memory for host DP matrices init.\n");
        ret = CL_OUT_OF_HOST_MEMORY;
        goto cleanup_host_alloc_dp;
    }

    M_matrix_host[0] = 0.0f;
    S_matrix_host[0] = 0.0f; 
    Ix_matrix_host[0] = NEGATIVE_INFINITY;
    Iy_matrix_host[0] = NEGATIVE_INFINITY;

    for (int i = 1; i <= seq1_len; ++i) {
        M_matrix_host[i * dp_cols] = NEGATIVE_INFINITY;
        Ix_matrix_host[i * dp_cols] = NEGATIVE_INFINITY;
        // penalty_open already includes one penalty_extend for the first char in gap
        Iy_matrix_host[i * dp_cols] = penalty_open + (i - 1) * penalty_extend;
        S_matrix_host[i * dp_cols] = headgp ? Iy_matrix_host[i*dp_cols] : 0.0f;
    }
    for (int j = 1; j <= seq2_len; ++j) {
        M_matrix_host[j] = NEGATIVE_INFINITY;
        Iy_matrix_host[j] = NEGATIVE_INFINITY;
        Ix_matrix_host[j] = penalty_open + (j - 1) * penalty_extend;
        S_matrix_host[j] = headgp ? Ix_matrix_host[j] : 0.0f;
    }

    ret = clEnqueueWriteBuffer(command_queue, M_matrix_gpu_buf, CL_TRUE, 0, dp_float_matrix_size_bytes, M_matrix_host, 0, NULL, NULL);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to write M_matrix init data. Error: %d\n", ret); goto cleanup_host_alloc_dp; }
    ret = clEnqueueWriteBuffer(command_queue, Ix_matrix_gpu_buf, CL_TRUE, 0, dp_float_matrix_size_bytes, Ix_matrix_host, 0, NULL, NULL);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to write Ix_matrix init data. Error: %d\n", ret); goto cleanup_host_alloc_dp; }
    ret = clEnqueueWriteBuffer(command_queue, Iy_matrix_gpu_buf, CL_TRUE, 0, dp_float_matrix_size_bytes, Iy_matrix_host, 0, NULL, NULL);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to write Iy_matrix init data. Error: %d\n", ret); goto cleanup_host_alloc_dp; }
    ret = clEnqueueWriteBuffer(command_queue, S_matrix_gpu_buf, CL_TRUE, 0, dp_float_matrix_size_bytes, S_matrix_host, 0, NULL, NULL);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to write S_matrix init data. Error: %d\n", ret); goto cleanup_host_alloc_dp; }

    // Create and build the program (moved after CPU memory allocation to allow cleanup on failure)
    program = clCreateProgramWithSource(context, 1, (const char **)&dp_kernel_source_str, NULL, &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create program with source. Error: %d\n", ret); goto cleanup_buffers; }

    ret = clBuildProgram(program, 1, &device_id, "-cl-std=CL1.2", NULL, NULL); // Specify OpenCL standard
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to build program. Error: %d\n", ret);
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        if(log) {
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "Build log:\n%s\n", log);
            free(log);
        }
        clReleaseProgram(program); program = NULL;
        goto cleanup_buffers;
    }

    // Create the kernel
    kernel = clCreateKernel(program, "dp_matrix_fill_kernel", &ret);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to create kernel. Error: %d\n", ret); if(program) clReleaseProgram(program); program = NULL; goto cleanup_buffers; }

    // 5. Set kernel arguments
    // Note: penalty_open for G__align11_cl is equivalent to (penalty_open_raw + penalty_extend) for the kernel.
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &seq1_gpu_buf);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &seq2_gpu_buf);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &scoring_matrix_gpu_buf);
    ret |= clSetKernelArg(kernel, 3, sizeof(int), &seq1_len);
    ret |= clSetKernelArg(kernel, 4, sizeof(int), &seq2_len);
    int nalphabets_stride_k = NALPHABETS_STRIDE;
    ret |= clSetKernelArg(kernel, 5, sizeof(int), &nalphabets_stride_k);
    ret |= clSetKernelArg(kernel, 6, sizeof(float), &penalty_open);      // This is penalty_open_extend for the kernel
    ret |= clSetKernelArg(kernel, 7, sizeof(float), &penalty_extend);
    ret |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &S_matrix_gpu_buf);
    ret |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &M_matrix_gpu_buf);
    ret |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &Ix_matrix_gpu_buf);
    ret |= clSetKernelArg(kernel, 11, sizeof(cl_mem), &Iy_matrix_gpu_buf);
    ret |= clSetKernelArg(kernel, 12, sizeof(cl_mem), &traceback_matrix_gpu_buf);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to set kernel arguments. Error: %d\n", ret); goto cleanup_kernel_program; }

    // 6. Host-side Loop for Kernel Launches (Anti-Diagonal Wavefront)
    cl_event prev_diag_marker_event = NULL;
    for (int k_sum = 2; k_sum <= seq1_len + seq2_len; ++k_sum) { // k_sum = i + j (1-based indices)
        int i_start = (k_sum > seq2_len + 1) ? (k_sum - seq2_len) : 1; // Corrected i_start for 1-based
        int i_end = (k_sum - 1 < seq1_len) ? (k_sum - 1) : seq1_len;   // Corrected i_end for 1-based
        
        int num_cells_on_diag = i_end - i_start + 1;
        if (num_cells_on_diag <= 0) continue;

        cl_event* kernel_events_for_diag = (cl_event*)malloc(num_cells_on_diag * sizeof(cl_event));
        if (!kernel_events_for_diag) { 
            fprintf(stderr, "Failed to allocate memory for kernel events list.\n"); 
            ret = CL_OUT_OF_HOST_MEMORY; goto cleanup_kernel_program; 
        }
        int current_event_idx = 0;

        for (int i_kernel = i_start; i_kernel <= i_end; ++i_kernel) {
            int j_kernel = k_sum - i_kernel;
            if (j_kernel < 1 || j_kernel > seq2_len) continue; // Ensure j_kernel is valid

            size_t global_offset[2] = { (size_t)i_kernel - 1, (size_t)j_kernel - 1 }; // Kernel uses get_global_id(0) for i-1
            size_t current_global_work_size[2] = { 1, 1 }; 
            
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, global_offset, 
                                         current_global_work_size, NULL, 
                                         (prev_diag_marker_event ? 1 : 0), 
                                         (prev_diag_marker_event ? &prev_diag_marker_event : NULL), 
                                         &kernel_events_for_diag[current_event_idx]);
            if (ret != CL_SUCCESS) { 
                fprintf(stderr, "Failed to enqueue kernel for cell (%d,%d) in diag k_sum=%d. Error: %d\n", i_kernel, j_kernel, k_sum, ret); 
                free(kernel_events_for_diag);
                goto cleanup_kernel_program; 
            }
            current_event_idx++;
        }

        if (prev_diag_marker_event) {
            clReleaseEvent(prev_diag_marker_event);
            prev_diag_marker_event = NULL;
        }

        if (current_event_idx > 0) { // Only if kernels were actually launched for this diagonal
            ret = clEnqueueMarkerWithWaitList(command_queue, current_event_idx, kernel_events_for_diag, &prev_diag_marker_event);
            for(int ev_idx = 0; ev_idx < current_event_idx; ++ev_idx) clReleaseEvent(kernel_events_for_diag[ev_idx]);
            free(kernel_events_for_diag);
            if (ret != CL_SUCCESS) { 
                fprintf(stderr, "Failed to enqueue marker for diag k_sum=%d. Error: %d\n", k_sum, ret); 
                goto cleanup_kernel_program; 
            }
        } else {
            free(kernel_events_for_diag); // No kernels launched, just free the list
        }
    }
    if (prev_diag_marker_event) clReleaseEvent(prev_diag_marker_event);

    // 7. Data Transfer (GPU to CPU)
    traceback_matrix_host = (int *)malloc(traceback_matrix_size_bytes);
    if (!traceback_matrix_host) { 
        fprintf(stderr, "Failed to allocate memory for host traceback matrix.\n"); 
        ret = CL_OUT_OF_HOST_MEMORY; goto cleanup_kernel_program; 
    }
    ret = clEnqueueReadBuffer(command_queue, traceback_matrix_gpu_buf, CL_TRUE, 0, traceback_matrix_size_bytes, traceback_matrix_host, 0, NULL, NULL);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to read traceback matrix from GPU. Error: %d\n", ret); goto cleanup_kernel_program; }

    float final_score_gpu_val;
    size_t final_score_offset_bytes = (seq1_len * dp_cols + seq2_len) * sizeof(float);
    ret = clEnqueueReadBuffer(command_queue, S_matrix_gpu_buf, CL_TRUE, final_score_offset_bytes, sizeof(float), &final_score_gpu_val, 0, NULL, NULL);
    if (ret != CL_SUCCESS) { fprintf(stderr, "Failed to read final score from GPU. Error: %d\n", ret); goto cleanup_kernel_program; }
    score = (double)final_score_gpu_val;

    // 8. CPU-based Traceback
    int current_i = seq1_len;
    int current_j = seq2_len;
    int align_buf_idx = buffer_size - 1; 
    aligned_seq1_out[align_buf_idx] = '\0';
    aligned_seq2_out[align_buf_idx] = '\0';
    align_buf_idx--;

    while (current_i > 0 || current_j > 0) {
        if (align_buf_idx < 0) { 
            fprintf(stderr, "Traceback exceeded buffer size. Required: %d, Provided: %d\n", (seq1_len + seq2_len), buffer_size);
            ret = CL_INVALID_VALUE; // Indicate error due to buffer size
            goto cleanup_kernel_program; 
        }
        int trace_dir = traceback_matrix_host[current_i * dp_cols + current_j];
        if (trace_dir == TRACE_DIAG && current_i > 0 && current_j > 0) {
            aligned_seq1_out[align_buf_idx] = seq1_str[current_i - 1];
            aligned_seq2_out[align_buf_idx] = seq2_str[current_j - 1];
            current_i--;
            current_j--;
        } else if (trace_dir == TRACE_UP && current_i > 0) { 
            aligned_seq1_out[align_buf_idx] = seq1_str[current_i - 1];
            aligned_seq2_out[align_buf_idx] = '-';
            current_i--;
        } else if (trace_dir == TRACE_LEFT && current_j > 0) { // TRACE_LEFT
            aligned_seq1_out[align_buf_idx] = '-';
            aligned_seq2_out[align_buf_idx] = seq2_str[current_j - 1];
            current_j--;
        } else { // Should not happen if current_i > 0 || current_j > 0
             // If current_i or current_j is 0, force gap until both are 0
            if(current_i > 0) {
                aligned_seq1_out[align_buf_idx] = seq1_str[current_i - 1];
                aligned_seq2_out[align_buf_idx] = '-';
                current_i--;
            } else if (current_j > 0) {
                aligned_seq1_out[align_buf_idx] = '-';
                aligned_seq2_out[align_buf_idx] = seq2_str[current_j - 1];
                current_j--;
            } else { // Both are 0, break
                break;
            }
        }
        align_buf_idx--;
    }
    // Shift alignment to beginning of buffers
    int start_idx_in_buf = align_buf_idx + 1;
    if (start_idx_in_buf < buffer_size) { // Check if there's anything to move
        memmove(aligned_seq1_out, &aligned_seq1_out[start_idx_in_buf], strlen(&aligned_seq1_out[start_idx_in_buf]) + 1);
        memmove(aligned_seq2_out, &aligned_seq2_out[start_idx_in_buf], strlen(&aligned_seq2_out[start_idx_in_buf]) + 1);
    } else if (buffer_size > 0) { // If start_idx_in_buf is at or beyond buffer_size, means empty alignment or error
        aligned_seq1_out[0] = '\0';
        aligned_seq2_out[0] = '\0';
    }

    // Set ret to CL_SUCCESS if everything went well up to this point
    // if(ret == 0) ret = CL_SUCCESS; // If ret wasn't set by an error above. ret is already CL_SUCCESS if no error.

cleanup_kernel_program:
    if(kernel) clReleaseKernel(kernel);
    if(program) clReleaseProgram(program);
cleanup_buffers:
    if(seq1_gpu_buf) clReleaseMemObject(seq1_gpu_buf);
    if(seq2_gpu_buf) clReleaseMemObject(seq2_gpu_buf);
    if(scoring_matrix_gpu_buf) clReleaseMemObject(scoring_matrix_gpu_buf);
    if(S_matrix_gpu_buf) clReleaseMemObject(S_matrix_gpu_buf);
    if(M_matrix_gpu_buf) clReleaseMemObject(M_matrix_gpu_buf);
    if(Ix_matrix_gpu_buf) clReleaseMemObject(Ix_matrix_gpu_buf);
    if(Iy_matrix_gpu_buf) clReleaseMemObject(Iy_matrix_gpu_buf);
    if(traceback_matrix_gpu_buf) clReleaseMemObject(traceback_matrix_gpu_buf);
cleanup_host_alloc_dp:
    if(M_matrix_host) free(M_matrix_host);
    if(Ix_matrix_host) free(Ix_matrix_host);
    if(Iy_matrix_host) free(Iy_matrix_host);
    if(S_matrix_host) free(S_matrix_host);
    if(traceback_matrix_host) free(traceback_matrix_host);
cleanup_host_alloc:
    if(seq1_int_host) free(seq1_int_host);
    if(seq2_int_host) free(seq2_int_host);
    if(scoring_matrix_flat_host) free(scoring_matrix_flat_host);
cleanup_local_cl:
    if (local_cl_setup) {
        if(command_queue) clReleaseCommandQueue(command_queue);
        if(context) clReleaseContext(context);
    }

    if (ret != CL_SUCCESS) {
        // Specific error codes from OpenCL are negative. Positive could be our own or uninitialized.
        // Standard OpenCL error codes are negative integers.
        if (ret != CL_OUT_OF_HOST_MEMORY && ret != CL_INVALID_VALUE) { // Check if it's an OpenCL error code we haven't specifically handled for return value
             fprintf(stderr, "G__align11_cl finished with OpenCL error code: %d\n", ret);
        } // Otherwise, specific error message was already printed.
        return NEGATIVE_INFINITY; // Indicate error
    }
    return score;
}
