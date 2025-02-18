@echo off
set INPUT_DIR=bagging_configs_grid_heart
set OUTPUT_DIR=results/bagging_configs_grid_heart_output
set PYTHON_SCRIPT=train.py
set HYPERPARAMETERS_FILE=cfg\best_decision_tree_heart.json

if not exist "%INPUT_DIR%" (
    echo Error: Input directory %INPUT_DIR% does not exist.
    exit /b 1
)

if not exist "%OUTPUT_DIR%" (
    echo Output directory %OUTPUT_DIR% does not exist. Creating it now...
    mkdir "%OUTPUT_DIR%"
)

for %%F in (%INPUT_DIR%\*) do (
    set "CONFIG_FILE=%%F"
    setlocal enabledelayedexpansion
    for %%A in ("!CONFIG_FILE!") do set "BASE_NAME=%%~nA"
    set "OUTPUT_FILE=%OUTPUT_DIR%\!BASE_NAME!_output.log"

    python "%PYTHON_SCRIPT%" -c "!CONFIG_FILE!" -p "%HYPERPARAMETERS_FILE%" > "!OUTPUT_FILE!" 2>&1

    echo Started processing !CONFIG_FILE!. Output is redirected to !OUTPUT_FILE!
    endlocal
)

echo All processes completed.
