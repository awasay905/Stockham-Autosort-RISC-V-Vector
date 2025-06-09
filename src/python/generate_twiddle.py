import math

# Define the size of the precomputed twiddle factor table
TWIDDLE_TABLE_SIZE = 8192

# Output file name
OUTPUT_FILENAME = "./src/assembly/twiddle_factors.s"

# Number of floating-point values per line in the assembly output for readability
FLOATS_PER_LINE = 8 

def precompute_twiddles():
    """
    Precomputes the twiddle factors based on the C code logic.
    Returns two lists: twiddle_re and twiddle_im.
    """
    twiddle_re = [0.0] * TWIDDLE_TABLE_SIZE
    twiddle_im = [0.0] * TWIDDLE_TABLE_SIZE

    for k in range(TWIDDLE_TABLE_SIZE):
        # The angles for the twiddle factors are scaled by PI / TWIDDLE_TABLE_SIZE
        # This means the table covers angles from 0 to almost PI.
        angle = float(k) * math.pi / TWIDDLE_TABLE_SIZE
        twiddle_re[k] = math.cos(angle)
        twiddle_im[k] = -math.sin(angle) # Store -sin(angle) for e^(-j*theta)
    return twiddle_re, twiddle_im

def generate_assembly_data(data_list, label, f):
    """
    Generates assembly .float directives for a given list of data.
    """
    f.write(f".global twiddle_real\n.global twiddle_imag\n")
    f.write(f".align 4\n")
    f.write(f"{label}:\n")
    
    for i in range(0, TWIDDLE_TABLE_SIZE, FLOATS_PER_LINE):
        chunk = data_list[i : i + FLOATS_PER_LINE]
        # Format each float to a reasonable precision (e.g., 10 decimal places)
        # and join them with commas.
        formatted_chunk = [f"{val:.10f}" for val in chunk]
        f.write(f"    .float " + ", ".join(formatted_chunk) + "\n")

if __name__ == "__main__":
    twiddle_re, twiddle_im = precompute_twiddles()

    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(".data\n") # Start of the data section

        print(f"Generating twiddle_real data ({TWIDDLE_TABLE_SIZE} floats)...")
        generate_assembly_data(twiddle_re, "twiddle_real", f)
        
        f.write("\n") # Add a newline for separation

        print(f"Generating twiddle_imag data ({TWIDDLE_TABLE_SIZE} floats)...")
        generate_assembly_data(twiddle_im, "twiddle_imag", f)

    print(f"\nAssembly twiddle factors saved to {OUTPUT_FILENAME}")
    print(f"Total size: {TWIDDLE_TABLE_SIZE * 4 * 2} bytes (for {TWIDDLE_TABLE_SIZE} complex numbers)")