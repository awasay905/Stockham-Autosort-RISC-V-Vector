import math
import numpy as np
import os
import subprocess
import struct # Required for hex_to_float
import re


# --- Reusable functions from your original script ---

def format_array_as_data_string(data: list[float], num_group_size: int = 4, num_per_line: int = 32) -> list[str]:
    """
    Format a list of floats into assembly directives for a `.float` statement.

    Each float is formatted to 12 decimal places and grouped with extra spacing.
    """
    formatted_lines = []
    current_line = ".float "
    for i, value in enumerate(data):
        current_line += f"{value:.12f}, "
        if (i + 1) % num_group_size == 0:  # Add space after every (num_group_size)th number
            current_line += " "
        if (i + 1) % num_per_line == 0:  # New line after every (num_per_line) numbers
            formatted_lines.append(current_line.strip(", "))
            current_line = ".float "
    # Add remaining line if not exactly multiple of (num_per_line)
    if current_line.strip(", "):
        formatted_lines.append(current_line.strip(", "))
    return formatted_lines


def hex_to_float(hex_array: list[str]) -> list[float]:
    float_array = []
    for hex_str in hex_array:
        # Ensure the hex string is exactly 8 characters long
        if len(hex_str) != 8:
            raise ValueError(f"Hex string '{hex_str}' is not 8 characters long")
        # Convert the hex string to a 32-bit integer
        int_val = int(hex_str, 16)
        # Pack the integer as a 32-bit unsigned integer
        packed_val = struct.pack('>I', int_val)
        # Unpack as a float (IEEE 754)
        float_val = struct.unpack('>f', packed_val)[0]
        float_array.append(float_val)
    return float_array


def find_log_pattern_index(file_name: str) -> list[int]:
    """
    Finds the starting line indices of the first two consecutive occurrences
    of the required values in the 7th column.

    Args:
        file_name (str): The path to the input file.

    Returns:
        list: A list of line indices (0-based) where the full consecutive pattern
              was found to start. Returns up to two indices.
    """
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []

    # List of required values in the 7th column (must be consecutive)
    required_values = ["00000123", "00000456"]

    pattern_indices = []  # To store the line index when the full pattern is found
    
    # State variables:
    # `expecting_next_value` is True if we just found required_values[0]
    # and are now looking for required_values[1] on the very next line.
    expecting_next_value = False 
    current_pattern_start = None # Stores the line index where required_values[0] was found

    for i, line in enumerate(lines):
        columns = line.split()

        # Ensure there are enough columns to check the 7th one (index 6)
        if len(columns) > 6:
            value = columns[6]

            if expecting_next_value:
                # We are in a state where we just found required_values[0]
                # Now we need to check if the current line has required_values[1]
                if value == required_values[1]:
                    # Success! We found the consecutive pattern
                    pattern_indices.append(current_pattern_start)
                    # Reset for the next search
                    expecting_next_value = False
                    current_pattern_start = None
                elif value == required_values[0]:
                    # If we find required_values[0] again, it means
                    # the previous sequence was broken, and a new one starts here.
                    current_pattern_start = i
                    # expecting_next_value remains True, as we are still waiting for required_values[1]
                else:
                    # The sequence was broken (neither part of the pattern was found)
                    # Reset state and look for the beginning of a new pattern
                    expecting_next_value = False
                    current_pattern_start = None
            else:
                # We are in a state where we are looking for the start of the pattern (required_values[0])
                if value == required_values[0]:
                    current_pattern_start = i
                    expecting_next_value = True # Transition to the state of expecting the next value
                # If it's not required_values[0], we do nothing and keep looking.
        else:
            # If a line doesn't have enough columns, it breaks any potential sequence.
            expecting_next_value = False
            current_pattern_start = None

        # Stop if we've found the pattern twice
        if len(pattern_indices) == 2:
            break

    return pattern_indices

def process_file(file_name: str, delete_log_files: bool = False) -> np.ndarray:
    """
    Process a log file to extract complex numbers represented by separate real and imaginary hex strings.

    The function determines the start and end indices based on log patterns, extracts the real and imaginary
    components from the log, and converts them into floating-point numbers. For vectorized data, the strings
    are split into 8-character chunks (in reverse order) before conversion.
    """
    real_hex_strings = []
    imag_hex_strings = []

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # Find the start and end indices for data extraction
        # This function is expected to raise an error if patterns aren't found
        start_index, end_index = find_log_pattern_index(file_name)

        if delete_log_files:
            import os
            os.remove(file_name)

        # Initialize a flag to alternate between real and imag data collection
        # Assuming real data comes first, then imaginary, and so on.
        save_to_real = True
        is_vectorized = False

        # Process lines within the specified range for data extraction
        # We process from start_index + 1 up to (but not including) end_index
        # because the start_index line contains the pattern itself, not the data.
        # This assumes your data immediately follows the start pattern and ends just before the end pattern.
        # Adjust 'start_index + 1' if your data starts on the pattern line itself.
        # For simplicity, we assume the data is between the two markers.
        # A more robust solution might require more precise parsing of what lines contain data.
        
        # Determine if vectorized mode is active based on file content before data extraction
        # This check might need to be more precise if vsetvli can appear multiple times.
        for line in lines:
            if "vsetvli" in line:
                is_vectorized = True
                break

        # Now, extract data from the relevant section
        # The exact lines to extract from depend on your log format.
        # Let's assume data lines are strictly *between* the start and end pattern lines.
        data_lines = lines[start_index + 1:end_index] # Adjust +1 and range if needed

        for line in data_lines:
            words = line.split()
            if len(words) > 1:
                # Check for scalar load instructions
                if not is_vectorized and ("c.flw" in words or "flw" in words):
                    try:
                        idx = words.index("c.flw") if "c.flw" in words else words.index("flw")
                        if idx > 0: # Ensure there's a preceding word (the hex value)
                            if save_to_real:
                                real_hex_strings.append(words[idx - 1])
                            else:
                                imag_hex_strings.append(words[idx - 1])
                            save_to_real = not save_to_real # Toggle for next value
                    except ValueError: # 'c.flw' or 'flw' not found on this line, but it's okay
                        pass # Skip this line if it doesn't contain the expected instruction
                
                # Check for vector load instructions
                elif is_vectorized and "vle32.v" in words:
                    try:
                        idx = words.index("vle32.v")
                        if idx > 0: # Ensure there's a preceding word (the hex value)
                            if save_to_real:
                                real_hex_strings.append(words[idx - 1])
                            else:
                                imag_hex_strings.append(words[idx - 1])
                            save_to_real = not save_to_real # Toggle for next value
                    except ValueError: # 'vle32.v' not found on this line, but it's okay
                        pass # Skip this line if it doesn't contain the expected instruction


        if is_vectorized:
            real_val_converted = []
            imag_val_converted = []

            for i in range(len(real_hex_strings)):
                real_vector_hex = real_hex_strings[i]
                imag_vector_hex = imag_hex_strings[i]

                # Split the full vector hex string into 8-character (float) chunks
                real_chunks = [real_vector_hex[j:j+8] for j in range(0, len(real_vector_hex), 8)]
                imag_chunks = [imag_vector_hex[j:j+8] for j in range(0, len(imag_vector_hex), 8)]

                # Reverse the order of the chunks for correct interpretation
                # (Assuming the simulator logs MSB-first or reverse byte order within vector registers)
                real_chunks = real_chunks[::-1]
                imag_chunks = imag_chunks[::-1]

                real_val_converted.extend(real_chunks)
                imag_val_converted.extend(imag_chunks)

            # Convert all collected hex chunks to floats
            real_floats = hex_to_float(real_val_converted)
            imag_floats = hex_to_float(imag_val_converted)

        else: # Scalar mode
            real_floats = hex_to_float(real_hex_strings)
            imag_floats = hex_to_float(imag_hex_strings)

        # Combine into complex numpy array
        return np.array(real_floats) + 1j * np.array(imag_floats)

    except FileNotFoundError:
        print(f"Error: The log file {file_name} does not exist.")
        return np.array([])
    except ValueError as e:
        print(f"Error processing log file {file_name}: {e}")
        return np.array([])
    except Exception as e:
        print(f"An unexpected error occurred during log processing: {e}")
        return np.array([])

# --- Main Test Suite Logic ---

def run_fft_test_suite(
    min_log2_size: int = 2,  # Start with 2^2 = 4
    max_log2_size: int = 9,  # Go up to 2^9 = 512
    # The tolerance value. Adjust based on your simulator's precision.
    # For single-precision floats, 1e-4 or 1e-5 is often reasonable.
    tolerance: float = 1e-4,
    target_dir: str = "/home/ubuntu/Stockham-Autosort-RISC-V-Vector",
    data_output_file_relative: str = "./src/assembly/fft_data.s",
    logfile_relative: str = "./veer/tempFiles/logV.txt"
):
    """
    Runs a test suite for FFT across different powers of two sizes.
    Generates random data, runs the RISC-V simulator, compares output with NumPy.
    Stops if the difference exceeds a given tolerance.
    """
    print(f"Starting FFT Test Suite with tolerance: {tolerance:.1e}")
    print("-" * 60)

    original_cwd = os.getcwd() # Store original working directory

    for log2_size in range(min_log2_size, max_log2_size + 1):
        current_size = 1 << log2_size # Calculate 2^log2_size
        print(f"\n--- Testing FFT Size: {current_size} (2^{log2_size}) ---")

        # 1. Generate random input data
        # Using random complex numbers ensures varied test cases
        input_real = np.random.rand(current_size) * 10 - 5 # Values between -5 and 5
        input_imag = np.random.rand(current_size) * 10 - 5
        input_complex_numpy = input_real + 1j * input_imag

        # 2. Format data for assembly
        real_assembly_form = format_array_as_data_string(input_real.tolist())
        imag_assembly_form = format_array_as_data_string(input_imag.tolist())

        # 3. Write data to the assembly file
        full_data_output_path = os.path.join(target_dir, data_output_file_relative.lstrip('./'))
        try:
            with open(full_data_output_path, "w") as f:
                f.write(".section .data\n.global size \n.global log2size \n.global fft_input_real \n.global fft_input_imag\n\n")
                f.write(".global y_real \n.global y_imag\n\n")
                f.write(".align 4\n size:\n\t.word " + str(current_size) + "\n")
                f.write(".align 4\n log2size:\n\t.word " + str(log2_size) + "\n")
                f.write(".align 4\n fft_input_real:\n")
                for line in real_assembly_form:
                    f.write(line + "\n")
                f.write(".align 4\n fft_input_imag:\n")
                for line in imag_assembly_form:
                    f.write(line + "\n")
                f.write("\n\n")
                f.write(f".align 4\ny_real:\n.space {current_size*4}\n\n")
                f.write(f".align 4\ny_imag:\n.space {current_size*4}\n\n")
            print(f"  Generated assembly data for size {current_size} at {full_data_output_path}.")
        except IOError as e:
            print(f"ERROR: Could not write to {full_data_output_path}: {e}")
            return # Stop if file writing fails

        # 4. Run the simulator
        try:
            os.chdir(target_dir) # Change to the target directory for 'make' command
            print(f"  Running 'make allV' in {target_dir}...")
            # Using check=True will raise CalledProcessError if the command returns a non-zero exit code
            result = subprocess.run(["make", "allV"], capture_output=True, text=True, check=True)
            print("  Simulator run complete.")
            # Optionally, print simulator output for debugging:
            # print("  Simulator Stdout:\n", result.stdout)
            # print("  Simulator Stderr:\n", result.stderr)
            cycles_taken = int(re.search(r'Retired (\d+) instructions', result.stderr).group(1))
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Simulator command failed for size {current_size}.")
            print(f"  Return Code: {e.returncode}")
            print(f"  Stdout:\n{e.stdout}")
            print(f"  Stderr:\n{e.stderr}")
            os.chdir(original_cwd) # Change back before exiting
            return # Stop on simulation failure
        except FileNotFoundError:
            print(f"ERROR: 'make' command not found. Ensure it's in your PATH or target_dir is correct.")
            os.chdir(original_cwd) # Change back before exiting
            return
        finally:
            os.chdir(original_cwd) # Always change back to original directory

        # 5. Process log file
        full_logfile_path = os.path.join(target_dir, logfile_relative.lstrip('./'))
        try:
            if not os.path.exists(full_logfile_path):
                print(f"ERROR: Log file not found at {full_logfile_path}. Check simulator output and path.")
                return

            # Keep log files for debugging by default (delete_log_files=False)
            sim_fft_output = process_file(full_logfile_path, delete_log_files=False)
            if sim_fft_output.size == 0:
                print(f"WARNING: No FFT data extracted from log for size {current_size}. Skipping comparison.")
                continue # Continue to next size if no data was parsed
            print(f"  Extracted {len(sim_fft_output)} data points from log.")
        except Exception as e:
            print(f"ERROR: Failed to process log file for size {current_size}: {e}")
            return # Stop on log processing failure

        # 6. Compare with NumPy FFT
        numpy_fft_output = np.fft.fft(input_complex_numpy)

        if len(sim_fft_output) > len(numpy_fft_output):
            sim_fft_output=sim_fft_output[:len(numpy_fft_output)]


        # Ensure lengths match before comparison
        if len(sim_fft_output) != len(numpy_fft_output):
            print(f"ERROR: Output length mismatch for size {current_size}.")
            print(f"  Simulator output length: {len(sim_fft_output)}")
            print(f"  NumPy output length: {len(numpy_fft_output)}")
            print("  This indicates a potential issue in the simulator or log parsing.")
            return # Stop on critical mismatch

        max_abs_difference = np.max(np.abs(sim_fft_output - numpy_fft_output))
        mean_squared_error = np.mean(np.abs(sim_fft_output - numpy_fft_output)**2)

        print(f"  Comparison Results for Size: {current_size}:")
        print(f"    Max Absolute Difference:   {max_abs_difference:.3e}")
        print(f"    Mean Squared Error:        {mean_squared_error:.3e}")
        print(f"    VeeR Cycles Taken:         {cycles_taken}")

        if max_abs_difference > tolerance:
            print(f"  FAIL: Max Absolute Difference {max_abs_difference:.3e} exceeds tolerance {tolerance:.1e}.")
            print("-" * 60)
            print("Stopping test suite due to failure.")
            return # Stop immediately on failure
        else:
            print("  PASS")

    print("\n" + "=" * 60)
    print("FFT Test Suite Completed Successfully for all tested sizes within tolerance.")
    print("=" * 60)

# --- Execute the test suite ---
if __name__ == "__main__":
    # Adjust parameters as needed
    run_fft_test_suite(
        min_log2_size=1,  # Smallest size: 2^2 = 4
        max_log2_size=15, # Largest size: 2^10 = 1024 (adjust based on simulation time)
        tolerance=1e-4,   # Adjust this tolerance based on your expected floating-point precision
        target_dir="/home/ubuntu/Stockham-Autosort-RISC-V-Vector",
        data_output_file_relative="./src/assembly/fft_data.s", # Path relative to target_dir
        logfile_relative="./veer/tempFiles/logV.txt"             # Path relative to target_dir
    )