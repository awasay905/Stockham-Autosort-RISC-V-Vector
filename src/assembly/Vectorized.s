#define STDOUT 0xd0580000

.section .text
.global _start
_start:
## START YOUR CODE HERE
    lw a0, size             # Load size of FFT from data section
    lw a1, log2size         # Load log2 of size from data section
    la a2, fft_input_real   # Load base address of input data real
    la a3, fft_input_imag   # Load base address of input data imag
    la a4, y_real           # Load base address of output data real
    la a5, y_imag           # Load base address of output data imag
    la a6, twiddle_real     # Load base address of twiddle factors real
    la a7, twiddle_imag     # Load base address of twiddle factors imag
    call stockham_fft

    li a2, 8
    call printToLogVectorized

    j _finish


# Function: stockham_fft
# Implements the Stockham FFT algorithm. DIT. Assumes data is 
# segemnted into real and imaginary parts.
# Input:
#    - a0: Input size N 
#    - a1: Log2 of N
#    - a2: Base address of input data real
#    - a3: Base address of input data imag
#    - a4: Base address of y array real
#    - a5: Base address of y array imag
#    - a6: Base address of twiddle factors real
#    - a7: Base address of twiddle factors imag
# Returns:
#    - a0: Base address of output data real
#    - a1: Base address of output data imag
stockham_fft:

    li t0, 1                # s = 1 
    srai t1, a0, 1          # m = n/2

    li s0, 0                # s0 = stage = 0
    stage_loop:
        bge s0, a1, stage_loop_end

        li t2, 8192             # t2 = 8192, Twiddle table size
        div t2, t2, t0          # t2 = twiddle scale
        slli t2, t2, 2          # t2 = twiddle scale * 4 (for byte addressing) 

        li s1, 0                # s1 = p 
        inner_p_loop:
            bge s1, t1, inner_p_loop_end
            vsetvli t3, t0, e32, m1, ta, ma

            mul t4, t0, s1      # t4 = sp = s * p
            mul t5, t0, t1      # t5 = m * s
            add t5, t5, t4      # t5 = spm
            li s2, 0            # q = 0
            innermost_q_loop:
                bge s2, t0, innermost_q_loop_end

                # s3 and s4 will be idx_a and idx_b
                add s3, s2, t4
                add s4, s2, t5
                slli s3, s3, 2
                slli s4, s4, 2

                # Calculte twiddle base
                mul s5, s2, t2              # q_start * twiddle * 4. t2 has twiddle sacle*4, s2 has curr q
                add s6, a6, s5
                add s7, a7, s5
                # Load twiddle factors
                vlse32.v v4, 0(s6), t2    # wq_real;
                vlse32.v v8, 0(s7), t2    # wq_imag;

                # Load input data from idx a
                add s6, s3, a2   # a2 is curr x base
                add s7, s3, a3  # a2 is curr x imag
                vle32.v v12, 0(s6)  # a_real
                vle32.v v16, 0(s7)  # a_imag;

                # Load input data from idx b
                add s6, s4, a2   # a2 is curr x base
                add s7, s4, a3  # a2 is curr x imag
                vle32.v v20, 0(s6)  # b_real_raw;
                vle32.v v24, 0(s7)  # b_imag_raw;



                vfmul.vv v28, v24, v8
                vfmul.vv v8, v20, v8
                vfmsac.vv v28, v20, v4   # b_wq_re
                vfmacc.vv v8, v24, v4    # b_wq_im

                vfadd.vv v4, v12, v28
                vfadd.vv v20, v16, v8

                vfsub.vv v24, v12, v28
                vfsub.vv v12, v16, v8

                # Calculate output index s3 and s4 will have y0 and y1
                add s3, s2, t4   
                add s3, s3, t4    # s3 = q + 2*sp
                add s4, s3, t0    # s4 = q + 2*sp + s
                slli s3, s3, 2
                slli s4, s4, 2

                add s5, s3, a4
                add s6, s3, a5
                # now s3 have y_re y0
                vse32.v v4, 0(s5)
                vse32.v v20, 0(s6)

                add s5, s4, a4 
                add s6, s4, a5
                vse32.v v24, 0(s5)
                vse32.v v12, 0(s6)
                

                add s2, s2, t3  # Increment q by VLEN
                j innermost_q_loop
            innermost_q_loop_end:

            addi s1, s1, 1
            j inner_p_loop
        inner_p_loop_end:
        

        srai t1, t1, 1          # m = m/2
        slli t0, t0, 1          # s = s*2
        addi s0, s0, 1          # Increment stage
        # Swap Pointers
        mv s7, a2 
        mv a2, a4
        mv a4, s7
        mv s7, a3
        mv a3, a5
        mv a5, s7               
        j stage_loop
    stage_loop_end:



    mv a0, a2                # Set output base address real
    mv a1, a3                # Set output base address imag
    ret


## END YOU CODE HERE

# Function: print
# Logs values from array in a0 into registers v1 for debugging and output.
# from using the log file.
# Inputs:
#   - a0: Base address of array real
#   - a1: Base address of array imag
#   - a2: Size of array
# Clobbers: t0,t1, t2,t3 ft0, ft1.
printToLogVectorized:        
    addi sp, sp, -4
    sw a0, 0(sp)

    li t0, 0x123                 # Pattern for help in python script
    li t0, 0x456                 # Pattern for help in python script
	li t0, 0		                # load i = 0
    printloop:
        vsetvli t3, a2, e32           # Set VLEN based on a2
        slli t4, t3, 2                # Compute VLEN * 4 for address increment

        vle32.v v1, (a0)              # Load real[i] into v1
        vle32.v v1, (a1)              # Load real[i] into v1
        add a0, a0, t4                # Increment pointer for real[] by VLEN * 4
        add a1, a1, t4                # Increment pointer for real[] by VLEN * 4
        add t0, t0, t3                # Increment index

        bge t0, a2, endPrintLoop      # Exit loop if i >= size
        j printloop                   # Jump to start of loop
    endPrintLoop:
    li t0, 0x123                    # Pattern for help in python script
    li t0, 0x456                    # Pattern for help in python script
	
    lw a0, 0(sp)
    addi sp, sp, 4

	jr ra


# Function: _finish
# VeeR Related function which writes to to_host which stops the simulator
_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish

    .rept 100
        nop
    .endr

.section .data

.align 4
y_real:
    .space 1024  # Reserve space for output data real

.align 4
y_imag:
    .space 1024  # Reserve space for output data imag