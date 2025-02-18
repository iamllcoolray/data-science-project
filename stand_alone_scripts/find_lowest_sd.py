import os
import re
import ast  # To safely parse dictionary-like strings

# Function to read the log file
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Regular expressions to capture the relevant sections
hyperparameter_pattern = r"Decision Tree\s*\{([^}]+)\}"
stdev_pattern = r"STDEV\s*\{([^}]+)\}"

# Function to extract hyperparameters and STDEV values
def extract_data(log_content):
    hyperparameters = re.findall(hyperparameter_pattern, log_content)
    stdevs = re.findall(stdev_pattern, log_content)
    return hyperparameters, stdevs

# Function to parse the dictionary-like string and extract values
def parse_dict(dict_str):
    try:
        return ast.literal_eval(f"{{{dict_str}}}")
    except (ValueError, SyntaxError):
        return {}

# Function to find the lowest mean_absolute_error in STDEV and corresponding hyperparameters in a single file
def find_lowest_mae_in_stdev(file_path):
    log_content = read_log_file(file_path)
    hyperparameters, stdevs = extract_data(log_content)

    lowest_mae = float("inf")
    best_hyperparameters = None

    for idx, stdev_str in enumerate(stdevs):
        stdev_data = parse_dict(stdev_str)
        mae = stdev_data.get("mean_absolute_error", None)

        if mae is not None and mae < lowest_mae:
            lowest_mae = mae
            best_hyperparameters = hyperparameters[idx]

    return lowest_mae, best_hyperparameters

# Function to iterate through subdirectories and summarize results
def process_p1_p2_heart(base_folder):
    folder_results = {}

    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path):
            log_files = [f for f in os.listdir(subdir_path) if f.endswith(".log")]

            if log_files:
                lowest_mae_in_folder = float("inf")
                best_hyperparameters_in_folder = None
                best_log_file = None

                for log_file in log_files:
                    log_file_path = os.path.join(subdir_path, log_file)
                    lowest_mae, best_hyperparameters = find_lowest_mae_in_stdev(log_file_path)

                    if lowest_mae < lowest_mae_in_folder:
                        lowest_mae_in_folder = lowest_mae
                        best_hyperparameters_in_folder = best_hyperparameters
                        best_log_file = log_file

                # Store the best result for the current folder
                folder_results[subdir] = (lowest_mae_in_folder, best_hyperparameters_in_folder, best_log_file)

    # Find the folder with the lowest mean_absolute_error across all folders
    lowest_mae_overall = float("inf")
    best_folder = None
    best_result = None

    for folder, (lowest_mae, hyperparameters, log_file) in folder_results.items():
        if lowest_mae < lowest_mae_overall:
            lowest_mae_overall = lowest_mae
            best_folder = folder
            best_result = (lowest_mae, hyperparameters, log_file)

    # Print the summary of results for each folder
    for folder, (lowest_mae, hyperparameters, log_file) in folder_results.items():
        print(f"Folder: {folder}")
        print(f"  Lowest MAE in STDEV: {lowest_mae}")
        print(f"  Corresponding Hyperparameters: {hyperparameters}")
        print(f"  Log File: {log_file}")
        print("-" * 40)

    # Print the folder with the lowest overall MAE
    if best_folder:
        print(f"\nFolder with Lowest Overall MAE in STDEV: {best_folder}")
        print(f"  Lowest MAE: {best_result[0]}")
        print(f"  Corresponding Hyperparameters: {best_result[1]}")
        print(f"  Log File: {best_result[2]}")

# Example usage
base_folder_path = r"C:\Users\friveran\Downloads\DataScienceFinalProject_v1\DataScienceFinalProject_v1\results"
process_p1_p2_heart(base_folder_path)
