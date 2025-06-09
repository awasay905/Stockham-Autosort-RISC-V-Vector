import math 
import numpy as np

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
    import struct
    float_array = []

    for hex_str in hex_array:
        # Ensure the hex string is exactly 8 characters long
        if len(hex_str) != 8:
            raise ValueError(
                f"Hex string '{hex_str}' is not 8 characters long")

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
    Locate the starting line indices where a specific hexadecimal log pattern occurs twice in a file.

    The function searches for the pattern defined by a set of required hexadecimal values.

    Parameters
    ----------
    file_name : str
        Path to the log file.

    Returns
    -------
    list of int
        A list containing up to two line indices where the pattern starts.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # List of required values in the 7th column
    required_values = ["00000123", "00000456"]
    found_values = []  # List to track when we find the required values
    pattern_indices = []  # To store the line index when the full pattern is found
    current_pattern_start = None  # To store where a potential pattern starts

    for i, line in enumerate(lines):
        columns = line.split()
        if len(columns) > 6:  # Check if there are enough columns
            value = columns[6]  # Get the 7th column (index 6)
            if value in required_values:
                if current_pattern_start is None and value == required_values[0]:
                    current_pattern_start = i  # Start tracking pattern from this line
                found_values.append(value)

            # If we found all the required values, save the index and reset
            if all(val in found_values for val in required_values):
                pattern_indices.append(current_pattern_start)
                found_values = []  # Reset for the next pattern
                current_pattern_start = None  # Reset pattern start

            # If we've found the pattern twice, we can stop
            if len(pattern_indices) == 2:
                break

    # Return the line indices where the pattern was found twice
    return pattern_indices




def process_file(file_name: str, delete_log_files: bool = False) -> np.ndarray:
    """
    Process a log file to extract complex numbers represented by separate real and imaginary hex strings.

    The function determines the start and end indices based on log patterns, extracts the real and imaginary
    components from the log, and converts them into floating-point numbers. For vectorized data, the strings
    are split into 8-character chunks (in reverse order) before conversion.
    """
    import numpy as np
    start_index, end_index = find_log_pattern_index(file_name)
    real = []
    imag = []

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()

        if delete_log_files:
            import os
            os.remove(file_name)

        # Ensure start and end indexes are within the valid range
        start_index = max(0, start_index)
        end_index = min(len(lines), end_index)

        # Initialize a flag to alternate between real and imag
        save_to_real = True
        is_vectorized = False

        # Process lines within the specified range
        for i in range(start_index, end_index):
            if not is_vectorized:
                if "vsetvli" in lines[i]:
                    is_vectorized = True
                    continue

                if "c.flw" in lines[i] or "flw" in lines[i]:
                    words = lines[i].split()
                    if len(words) > 1:
                        if "c.flw" in lines[i]:
                            index_of_cflw = words.index("c.flw")
                        else:
                            index_of_cflw = words.index("flw")
                        if index_of_cflw > 0:
                            if save_to_real:
                                real.append(words[index_of_cflw - 1])
                                save_to_real = False
                            else:
                                imag.append(words[index_of_cflw - 1])
                                save_to_real = True

            else:
                if "vle32.v" in lines[i]:
                    words = lines[i].split()
                    index_of_cflw = words.index("vle32.v")
                    if save_to_real:
                        real.append(words[index_of_cflw - 1])
                        save_to_real = False
                    else:
                        imag.append(words[index_of_cflw - 1])
                        save_to_real = True

        # return hex_to_float(real), hex_to_float(imag)
        if (is_vectorized):
            realVal = []
            imagVal = []

            for i in range(len(real)):
                realVector = real[i]
                imagVector = imag[i]

                # split the strings into 8 bit chunks
                realVector = [realVector[i:i+8]
                              for i in range(0, len(realVector), 8)]
                imagVector = [imagVector[i:i+8]
                              for i in range(0, len(imagVector), 8)]

                # reverse the order of the chunks
                realVector = realVector[::-1]
                imagVector = imagVector[::-1]

                realVal.extend(realVector)
                imagVal.extend(imagVector)

            real = realVal
            imag = imagVal
        return np.array(hex_to_float(real)) + 1j * np.array(hex_to_float(imag))

    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
        return real, imag




# change real/imag from here to any data you want
real = [-3.464429692824, -1.306968446289, 1.683990727203, 4.169108391309,  3.547545173651, 4.752043990383, -2.608885283517, -3.132968008724]
imag = [-0.361667264722, -4.287705449603, -3.017771235815, -2.836087560137,  2.932867944814, -4.406974029014, -2.736816297475, -4.534302847279]

real_assembly_form = format_array_as_data_string(real)
imag_assembly_form = format_array_as_data_string(imag)

data_output_file = "./src/assembly/fft_data.s"

# with open(data_output_file, "w") as f:
#     f.write(".section .data\n.global size \n.global log2size \n.global fft_input_real \n.global fft_input_imag\n\n")
#     f.write(".align 4\n size:\n\t.word " + str(len(real)) + "\n")
#     f.write(".align 4\n log2size:\n\t.word " + str(int(math.log2(len(real)))) + "\n")
#     f.write(".align 4\n fft_input_real:\n")
#     for i in range(len(real_assembly_form)):
#         f.write(real_assembly_form[i]+ "\n")
#     f.write(".align 4\n fft_input_imag:\n")
#     for i in range(len(imag_assembly_form)):
#         f.write(imag_assembly_form[i]+ "\n")


logfile = "./veer/tempFiles/logV.txt"

data  = process_file(logfile)

numpy_fft_data = np.fft.fft(np.array(real) + 1j * np.array(imag))
my_fft_to_compare = data.astype(np.complex64, copy=False)

print("Data from log file:")
print(my_fft_to_compare)
print("Numpy FFT data:")
print(numpy_fft_data)

max_abs_difference = np.max(np.abs(my_fft_to_compare - numpy_fft_data))
mean_squared_error = np.mean(np.abs(my_fft_to_compare - numpy_fft_data)**2)

print(f"    Max Absolute Difference: {max_abs_difference:.3e}")
print(f"    Mean Squared Error: {mean_squared_error:.3e}")
