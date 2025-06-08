#include <stdio.h>
#include <stdlib.h>
#include <math.h> // For cosf, sinf, M_PI

#ifndef M_PI
#define M_PI 3.14159265358979323846 // Define M_PI if not already defined
#endif

// Define the size of the precomputed twiddle factor table
// This table stores factors for angles up to PI (half a full circle), as needed by Stockham.
#define TWIDDLE_TABLE_SIZE 8192

// Global arrays for precomputed twiddle factors
// twiddle_re[k] = cos(k * PI / TWIDDLE_TABLE_SIZE)
// twiddle_im[k] = -sin(k * PI / TWIDDLE_TABLE_SIZE)  (Negative for e^(-j*theta))
float twiddle_re[TWIDDLE_TABLE_SIZE];
float twiddle_im[TWIDDLE_TABLE_SIZE];

/**
 * @brief Precomputes the twiddle factors for the Stockham FFT.
 * This function should be called once before performing any FFT computations.
 */
void precompute_twiddles() {
    for (int k = 0; k < TWIDDLE_TABLE_SIZE; ++k) {
        // The angles for the twiddle factors are scaled by PI / TWIDDLE_TABLE_SIZE
        // This means the table covers angles from 0 to almost PI.
        float angle = (float)k * M_PI / TWIDDLE_TABLE_SIZE;
        twiddle_re[k] = cosf(angle);
        twiddle_im[k] = -sinf(angle); // Store -sin(angle) for e^(-j*theta)
    }
}

/**
 * @brief Performs an out-of-place Stockham Fast Fourier Transform.
 *
 * @param n The size of the FFT (must be a power of 2).
 * @param x_re Pointer to the real part of the input array. This array will
 *             also contain the real part of the FFT result upon completion.
 * @param x_im Pointer to the imaginary part of the input array. This array will
 *             also contain the imaginary part of the FFT result upon completion.
 * @param y_re Pointer to the real part of an auxiliary (scratch) array of size n.
 * @param y_im Pointer to the imaginary part of an auxiliary (scratch) array of size n.
 *             These auxiliary arrays are used for out-of-place computations and
 *             should be allocated by the caller.
 */
void stockham_fft(int n, float* x_re, float* x_im, float* y_re, float* y_im) {
    // Calculate log2(n) for the number of stages
    int log2_n = 0;
    while ((1 << log2_n) < n) {
        log2_n++;
    }

    // Initial parameters for the first stage
    int s = 1;     // Stride for the 'q' loop (size of data within a butterfly group)
    int m = n / 2; // Number of 'p' blocks (half of current_n)

    // Pointers for current input and output buffers
    // These will swap roles at each stage
    float* current_x_re = x_re;
    float* current_x_im = x_im;
    float* current_y_re = y_re;
    float* current_y_im = y_im;

    for (int stage = 0; stage < log2_n; ++stage) {
        printf("Stage %d: m = %d, s = %d\n", stage, m, s);
        // Calculate the scaling factor to map 'q' to the precomputed twiddle table index.
        // The angle in the inner loop is `q * PI / s`.
        // The table is precomputed for `k * PI / TWIDDLE_TABLE_SIZE`.
        // So, we need `k = q * (TWIDDLE_TABLE_SIZE / s)`.
        int twiddle_scale = TWIDDLE_TABLE_SIZE / s;

        for (int p = 0; p < m; ++p) {
            int sp = s * p;
            int spm = sp + s*m;
            for (int q = 0; q < s; ++q) {
                // Calculate input indices for 'a' and 'b'
                // 'a' is from the first half of the current block
                // 'b' is from the second half, 'm' sub-blocks away
                int idx_a = q + sp;
                int idx_b = q + spm;

                // Get the twiddle factor from the precomputed table
                int tw_idx = q * twiddle_scale;
                float wq_re = twiddle_re[tw_idx];
                float wq_im = twiddle_im[tw_idx]; // This already contains the negative sign

                // Read input complex numbers 'a' and 'b'
                float a_re = current_x_re[idx_a];
                float a_im = current_x_im[idx_a];

                float b_re_raw = current_x_re[idx_b];
                float b_im_raw = current_x_im[idx_b];

                // Perform complex multiplication: b_final = b_raw * wq
                // (b_re_raw + j*b_im_raw) * (wq_re + j*wq_im)
                //  = (b_re_raw*wq_re - b_im_raw*wq_im) + j*(b_re_raw*wq_im + b_im_raw*wq_re)
                float b_wq_re = b_re_raw * wq_re - b_im_raw * wq_im;
                float b_wq_im = b_re_raw * wq_im + b_im_raw * wq_re;

                // Calculate output indices for y[q + s*(2*p + 0)] and y[q + s*(2*p + 1)]
                // This interleaves the results of the butterflies
                int idx_y0 = q + 2*sp; // For a + b_wq
                int idx_y1 = q + 2*sp + s; // For a - b_wq

                // Write results to the current output buffer 'y'
                current_y_re[idx_y0] = a_re + b_wq_re;
                current_y_im[idx_y0] = a_im + b_wq_im;

                current_y_re[idx_y1] = a_re - b_wq_re;
                current_y_im[idx_y1] = a_im - b_wq_im;
            }
        }

        // Update parameters for the next stage
        m = m / 2; // Half the number of 'p' blocks
        s = s * 2; // Double the 'q' stride (data group size)

        // Swap the roles of input and output buffers for the next stage
        // This is a crucial part of Stockham, making it out-of-place and efficient.
        float* temp_ptr_re = current_x_re;
        current_x_re = current_y_re;
        current_y_re = temp_ptr_re;

        float* temp_ptr_im = current_x_im;
        current_x_im = current_y_im;
        current_y_im = temp_ptr_im;
    }

    // After all stages, the final result might reside in the original 'x' buffer
    // or the 'y' auxiliary buffer, depending on whether log2_n is even or odd.
    // If 'current_x_re' does not point to the original 'x_re' (meaning the result is in 'y'),
    // we copy the result back to the original 'x' arrays.
    if (current_x_re != x_re) {
        for (int i = 0; i < n; ++i) {
            x_re[i] = current_x_re[i];
            x_im[i] = current_x_im[i];
        }
    }
}

// --- Example Usage ---
int main() {
    // 1. Precompute twiddle factors once at the start of your application
    precompute_twiddles();

    // Example FFT size (must be a power of 2, e.g., 8, 16, 32, 64, 128, etc.)
    int N = 8; 

    // Allocate memory for the input and auxiliary arrays
    // Caller is responsible for allocating and freeing these buffers.
    float* x_re = (float*)malloc(N * sizeof(float));
    float* x_im = (float*)malloc(N * sizeof(float));
    float* y_re = (float*)malloc(N * sizeof(float)); // Auxiliary buffer
    float* y_im = (float*)malloc(N * sizeof(float)); // Auxiliary buffer

    if (!x_re || !x_im || !y_re || !y_im) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // 2. Initialize input signal (e.g., a simple impulse: DFT is all ones)
    // x[0] = 1.0 + 0.0j, x[1..N-1] = 0.0 + 0.0j
    x_re[0] = 10.0f;
    x_im[0] = 20.0f;
    for (int i = 1; i < N; ++i) {
        x_re[i] = 1.0f;
        x_im[i] = 1.0f;
    }

    printf("Input signal (real, imag):\n");
    for (int i = 0; i < N; ++i) {
        printf("  (%f, %f)\n", x_re[i], x_im[i]);
    }

    // 3. Perform the FFT
    stockham_fft(N, x_re, x_im, y_re, y_im);

    printf("\nFFT Result (real, imag):\n");
    for (int i = 0; i < N; ++i) {
        printf("  (%f, %f)\n", x_re[i], x_im[i]);
    }

    // 4. Free allocated memory
    free(x_re);
    free(x_im);
    free(y_re);
    free(y_im);

    return 0;
}