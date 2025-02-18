import os
import re

# Function to read the log file
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Regular expressions to capture the relevant information
hyperparameter_pattern = r"Decision Tree\s*\{([^}]+)\}"
mean_pattern = r"MEAN\s*\{([^}]+)\}"

# Function to extract hyperparameters and mean values
def extract_data(log_content):
    hyperparameters = re.findall(hyperparameter_pattern, log_content)
    means = re.findall(mean_pattern, log_content)
    return hyperparameters, means

# Function to parse the mean dictionary and extract MAE
def parse_mean(mean_str):
    mean_dict = {}
    for line in mean_str.split(","):
        key, value = line.split(":")
        mean_dict[key.strip()] = float(value.strip())
    return mean_dict

# Function to find the lowest MAE and corresponding hyperparameters in a single file
def find_lowest_mae(file_path):
    log_content = read_log_file(file_path)
    hyperparameters, means = extract_data(log_content)

    lowest_mae = float("inf")
    best_hyperparameters = None

    for idx, mean_str in enumerate(means):
        mean_data = parse_mean(mean_str)
        mae = mean_data.get("'mean_absolute_error'", None)
        
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
                    lowest_mae, best_hyperparameters = find_lowest_mae(log_file_path)

                    if lowest_mae < lowest_mae_in_folder:
                        lowest_mae_in_folder = lowest_mae
                        best_hyperparameters_in_folder = best_hyperparameters
                        best_log_file = log_file

                # Store the best result for the current folder
                folder_results[subdir] = (lowest_mae_in_folder, best_hyperparameters_in_folder, best_log_file)

    # Find the folder with the lowest MAE across all folders
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
        print(f"  Lowest MAE: {lowest_mae}")
        print(f"  Corresponding Hyperparameters: {hyperparameters}")
        print(f"  Log File: {log_file}")
        print("-" * 40)

    # Print the folder with the lowest overall MAE
    if best_folder:
        print(f"\nFolder with Lowest Overall MAE: {best_folder}")
        print(f"  Lowest MAE: {best_result[0]}")
        print(f"  Corresponding Hyperparameters: {best_result[1]}")
        print(f"  Log File: {best_result[2]}")




base_folder_path = r"C:\Users\friveran\Downloads\DataScienceFinalProject_v1\DataScienceFinalProject_v1\results"
#"C:\path\to\folder"
process_p1_p2_heart(base_folder_path)
