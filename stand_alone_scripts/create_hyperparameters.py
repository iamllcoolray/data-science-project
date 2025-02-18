import os
import yaml

input_folder = "bagging_configs_grid_heart"
output_folder = "bagging_configs_grid_heart"

# Function to modify the YAML content
def modify_yaml_content(content):
    content["MODEL"]["ENSEMBLE_HYPERPARAMETERS"]["EPSILON"] = 0.01
    content["DATA"]["DATA_PATH"] = 'heart.csv'
    content["DATA"]["FEATURES"] = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    content["DATA"]["TARGET"] = 'target'

    return content

for i in range(11):
    input_file = os.path.join(input_folder, f"{i}.yaml")
    output_file = os.path.join(output_folder, f"{i}.yaml")

    with open(input_file, "r") as f:
        yaml_content = yaml.safe_load(f)

    modified_content = modify_yaml_content(yaml_content)

    with open(output_file, "w") as f:
        yaml.dump(modified_content, f)