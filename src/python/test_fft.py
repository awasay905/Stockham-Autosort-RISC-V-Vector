import math
import numpy as np
import os
import subprocess
import struct
import re
import json # Import for JSON operations
import datetime # Import for timestamps
import hashlib # Import for hashing


# --- Configuration ---
RESULTS_LOG_FILE = "fft_results_history.json" # Name of the JSON log file
VECTORIZED_ASM_FILE_RELATIVE = "./src/assembly/Vectorized.s" # Path to your assembly code


# --- Reusable functions (from your original script, slightly adjusted) ---

def format_array_as_data_string(data: list[float], num_group_size: int = 4, num_per_line: int = 32) -> list[str]:
    """
    Format a list of floats into assembly directives for a `.float` statement.
    Each float is formatted to 12 decimal places and grouped with extra spacing.
    """
    formatted_lines = []
    current_line = ".float "
    for i, value in enumerate(data):
        current_line += f"{value:.12f}, "
        if (i + 1) % num_group_size == 0:
            current_line += " "
        if (i + 1) % num_per_line == 0:
            formatted_lines.append(current_line.strip(", "))
            current_line = ".float "
    if current_line.strip(", "):
        formatted_lines.append(current_line.strip(", "))
    return formatted_lines


def hex_to_float(hex_array: list[str]) -> list[float]:
    float_array = []
    for hex_str in hex_array:
        if len(hex_str) != 8:
            raise ValueError(f"Hex string '{hex_str}' is not 8 characters long")
        int_val = int(hex_str, 16)
        packed_val = struct.pack('>I', int_val) # Assuming big-endian for hex strings, check your simulator's log output
        float_val = struct.unpack('>f', packed_val)[0]
        float_array.append(float_val)
    return float_array


def find_log_pattern_index(file_name: str) -> list[int]:
    """
    Finds the starting line indices of the first two consecutive occurrences
    of the required values in the 7th column.
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

    required_values = ["00000123", "00000456"]
    pattern_indices = []
    expecting_next_value = False
    current_pattern_start = None

    for i, line in enumerate(lines):
        columns = line.split()

        if len(columns) > 6:
            value = columns[6]

            if expecting_next_value:
                if value == required_values[1]:
                    pattern_indices.append(current_pattern_start)
                    expecting_next_value = False
                    current_pattern_start = None
                elif value == required_values[0]:
                    current_pattern_start = i
                else:
                    expecting_next_value = False
                    current_pattern_start = None
            else:
                if value == required_values[0]:
                    current_pattern_start = i
                    expecting_next_value = True
        else:
            expecting_next_value = False
            current_pattern_start = None

        if len(pattern_indices) == 2:
            break

    return pattern_indices

def process_file(file_name: str, delete_log_files: bool = False) -> np.ndarray:
    """
    Process a log file to extract complex numbers represented by separate real and imaginary hex strings.
    """
    real_hex_strings = []
    imag_hex_strings = []

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()

        start_indices = find_log_pattern_index(file_name)
        if len(start_indices) < 2:
            raise ValueError(f"Could not find both log patterns in {file_name}. Log may be incomplete or malformed.")
        
        start_data_index = start_indices[0] + 1
        end_data_index = start_indices[1] # Data ends *before* the second pattern

        if delete_log_files:
            os.remove(file_name)

        save_to_real = True
        is_vectorized = False
        
        # Check for vectorization (can be more robust if vsetvli is used frequently)
        for line in lines[start_data_index:end_data_index]:
            if "vsetvli" in line:
                is_vectorized = True
                break

        for line in lines[start_data_index:end_data_index]:
            words = line.split()
            if len(words) > 1:
                if not is_vectorized and ("c.flw" in words or "flw" in words):
                    try:
                        idx = words.index("c.flw") if "c.flw" in words else words.index("flw")
                        if idx > 0:
                            if save_to_real:
                                real_hex_strings.append(words[idx - 1])
                            else:
                                imag_hex_strings.append(words[idx - 1])
                            save_to_real = not save_to_real
                    except ValueError:
                        pass
                elif is_vectorized and "vle32.v" in words:
                    try:
                        idx = words.index("vle32.v")
                        if idx > 0:
                            if save_to_real:
                                real_hex_strings.append(words[idx - 1])
                            else:
                                imag_hex_strings.append(words[idx - 1])
                            save_to_real = not save_to_real
                    except ValueError:
                        pass

        if is_vectorized:
            real_val_converted = []
            imag_val_converted = []
            
            # Ensure real_hex_strings and imag_hex_strings are of same length
            # and that they actually contain hex strings from log.
            if len(real_hex_strings) == len(imag_hex_strings) and len(real_hex_strings) > 0:
                for i in range(len(real_hex_strings)):
                    real_vector_hex = real_hex_strings[i]
                    imag_vector_hex = imag_hex_strings[i]

                    # Split and reverse for correct interpretation
                    real_chunks = [real_vector_hex[j:j+8] for j in range(0, len(real_vector_hex), 8)][::-1]
                    imag_chunks = [imag_vector_hex[j:j+8] for j in range(0, len(imag_vector_hex), 8)][::-1]

                    real_val_converted.extend(real_chunks)
                    imag_val_converted.extend(imag_chunks)

                real_floats = hex_to_float(real_val_converted)
                imag_floats = hex_to_float(imag_val_converted)
            else:
                 print("WARNING: Vectorized mode active but insufficient or mismatched hex strings for extraction.")
                 real_floats = []
                 imag_floats = []

        else: # Scalar mode
            real_floats = hex_to_float(real_hex_strings)
            imag_floats = hex_to_float(imag_hex_strings)

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


# --- JSON History Management Functions ---

def load_fft_history(log_path: str) -> dict:
    """Loads existing FFT test history from a JSON file."""
    try:
        with open(log_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"runs": []}
    except json.JSONDecodeError:
        print(f"WARNING: Malformed JSON log file '{log_path}'. Starting with empty history.")
        return {"runs": []}

def save_fft_history(history: dict, log_path: str):
    """Saves FFT test history to a JSON file."""
    try:
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"  Test results saved to {log_path}")
    except IOError as e:
        print(f"ERROR: Could not save results to {log_path}: {e}")


def get_code_version_info(code_file_path: str) -> tuple[str, str]:
    """Reads the assembly code and returns its content and SHA256 hash."""
    try:
        with open(code_file_path, 'r') as f:
            code_content = f.read()
        code_hash = hashlib.sha256(code_content.encode()).hexdigest()
        return code_content, code_hash
    except FileNotFoundError:
        print(f"ERROR: Assembly code file '{code_file_path}' not found.")
        return "", ""
    except Exception as e:
        print(f"ERROR: Could not read assembly code file '{code_file_path}': {e}")
        return "", ""


# --- Main Test Suite Logic ---

def run_fft_test_suite(
    min_log2_size: int = 1,
    max_log2_size: int = 16,
    tolerance: float = 1e-4,
    target_dir: str = "/home/ubuntu/Stockham-Autosort-RISC-V-Vector",
    data_output_file_relative: str = "./src/assembly/fft_data.s",
    logfile_relative: str = "./veer/tempFiles/logV.txt"
):
    """
    Runs a test suite for FFT across different powers of two sizes.
    Generates random data, runs the RISC-V simulator, compares output with NumPy.
    Saves results to a JSON history file, tracking code versions.
    """
    print(f"Starting FFT Test Suite with tolerance: {tolerance:.1e}")
    print("-" * 60)

    original_cwd = os.getcwd()
    full_vectorized_asm_path = os.path.join(target_dir, VECTORIZED_ASM_FILE_RELATIVE.lstrip('./'))
    full_results_log_path = os.path.join(target_dir, RESULTS_LOG_FILE)

    # Get current code version information
    current_code_content, current_code_hash = get_code_version_info(full_vectorized_asm_path)
    if not current_code_content: # Handle case where file not found or error reading
        return

    # Initialize data structure for this run
    current_run_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "code_hash": current_code_hash,
        "code_content": current_code_content,
        "results_per_size": {},
        "overall_status": "PASS" # Assume pass until a failure occurs
    }

    test_failed = False # Flag to track overall suite status

    for log2_size in range(min_log2_size, max_log2_size + 1):
        current_size = 1 << log2_size
        print(f"\n--- Testing FFT Size: {current_size} (2^{log2_size}) ---")

        input_real = np.random.rand(current_size) * 10 - 5
        input_imag = np.random.rand(current_size) * 10 - 5
        input_complex_numpy = input_real + 1j * input_imag

        real_assembly_form = format_array_as_data_string(input_real.tolist())
        imag_assembly_form = format_array_as_data_string(input_imag.tolist())

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
            test_failed = True
            break

        try:
            os.chdir(target_dir)
            print(f"  Running 'make allV' in {target_dir}...")
            result = subprocess.run(["make", "allV"], capture_output=True, text=True, check=True)
            print("  Simulator run complete.")
            cycles_match = re.search(r'Retired (\d+) instructions', result.stderr)
            cycles_taken = int(cycles_match.group(1)) if cycles_match else -1 # Default to -1 if not found
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Simulator command failed for size {current_size}.")
            print(f"  Return Code: {e.returncode}")
            print(f"  Stdout:\n{e.stdout}")
            print(f"  Stderr:\n{e.stderr}")
            test_failed = True
            break
        except FileNotFoundError:
            print(f"ERROR: 'make' command not found. Ensure it's in your PATH or target_dir is correct.")
            test_failed = True
            break
        finally:
            os.chdir(original_cwd)

        full_logfile_path = os.path.join(target_dir, logfile_relative.lstrip('./'))
        try:
            if not os.path.exists(full_logfile_path):
                print(f"ERROR: Log file not found at {full_logfile_path}. Check simulator output and path.")
                test_failed = True
                break

            sim_fft_output = process_file(full_logfile_path, delete_log_files=False)
            if sim_fft_output.size == 0:
                print(f"WARNING: No FFT data extracted from log for size {current_size}. Skipping comparison.")
                test_failed = True # Consider this a failure if data cannot be extracted
                break
            print(f"  Extracted {len(sim_fft_output)} data points from log.")
        except Exception as e:
            print(f"ERROR: Failed to process log file for size {current_size}: {e}")
            test_failed = True
            break

        numpy_fft_output = (np.fft.fft(input_complex_numpy) / current_size).astype(np.complex64)

        if len(sim_fft_output) > len(numpy_fft_output):
            sim_fft_output = sim_fft_output[:len(numpy_fft_output)]

        if len(sim_fft_output) != len(numpy_fft_output):
            print(f"ERROR: Output length mismatch for size {current_size}.")
            print(f"  Simulator output length: {len(sim_fft_output)}")
            print(f"  NumPy output length: {len(numpy_fft_output)}")
            print("  This indicates a potential issue in the simulator or log parsing.")
            test_failed = True
            break

        max_abs_difference = np.max(np.abs(sim_fft_output - numpy_fft_output))
        mean_squared_error = np.mean(np.abs(sim_fft_output - numpy_fft_output)**2)

        result_status = "PASS"
        if max_abs_difference > tolerance:
            result_status = "FAIL"
            test_failed = True # Mark overall suite as failed

        # Store results for this size
        current_run_data["results_per_size"][str(current_size)] = {
            "max_abs_diff": float(max_abs_difference), # Convert numpy float to Python float for JSON
            "mean_squared_error": float(mean_squared_error),
            "cycles": cycles_taken,
            "status": result_status
        }

        print(f"  Comparison Results for Size: {current_size}:")
        print(f"    Max Absolute Difference:   {max_abs_difference:.3e}")
        print(f"    Mean Squared Error:        {mean_squared_error:.3e}")
        print(f"    VeeR Cycles Taken:         {cycles_taken}")

        if result_status == "FAIL":
            print(f"  FAIL: Max Absolute Difference {max_abs_difference:.3e} exceeds tolerance {tolerance:.1e}.")
            print("-" * 60)
            print("Stopping test suite due to failure.")
            break # Stop immediately on failure
        else:
            print("  PASS")

    if test_failed:
        current_run_data["overall_status"] = "FAIL"
        print("\n" + "=" * 60)
        print("FFT Test Suite FAILED.")
        print("=" * 60)
    else:
        current_run_data["overall_status"] = "PASS"
        print("\n" + "=" * 60)
        print("FFT Test Suite Completed Successfully for all tested sizes within tolerance.")
        print("=" * 60)

    # Load full history, update, and save
    history = load_fft_history(full_results_log_path)
    
    # Check if this code version already exists in history
    found_existing_entry = False
    for i, run in enumerate(history["runs"]):
        if run.get("code_hash") == current_code_hash:
            history["runs"][i] = current_run_data # Update existing entry
            found_existing_entry = True
            break
    
    if not found_existing_entry:
        history["runs"].append(current_run_data) # Add new entry
    
    save_fft_history(history, full_results_log_path)


# --- Execute the test suite ---
if __name__ == "__main__":
    run_fft_test_suite(
        min_log2_size=1,
        max_log2_size=16,
        tolerance=1e-4,
        target_dir="/home/ubuntu/Stockham-Autosort-RISC-V-Vector",
        data_output_file_relative="./src/assembly/fft_data.s",
        logfile_relative="./veer/tempFiles/logV.txt"
    )