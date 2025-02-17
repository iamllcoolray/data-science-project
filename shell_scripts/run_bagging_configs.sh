#!/bin/bash

INPUT_DIR="bagging_configs_grid_heart"  
OUTPUT_DIR="results/bagging_configs_grid_heart_output"
PYTHON_SCRIPT="train.py" 
HYPERPARAMETERS_FILE="cfg/best_decision_tree_heart.json"  

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory $INPUT_DIR does not exist."
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

for CONFIG_FILE in "$INPUT_DIR"/*; do
  BASE_NAME=$(basename "$CONFIG_FILE")

  OUTPUT_FILE="$OUTPUT_DIR/${BASE_NAME%.*}_output.log"

  python "$PYTHON_SCRIPT" -c "$CONFIG_FILE" -p "$HYPERPARAMETERS_FILE" > "$OUTPUT_FILE" 2>&1

  echo "Started processing $CONFIG_FILE. Output is redirected to $OUTPUT_FILE"
done

echo "All processes completed."
