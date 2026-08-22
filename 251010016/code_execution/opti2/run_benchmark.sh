#!/bin/bash

# Define the Python script name
PYTHON_SCRIPT="FDM_WARP_test_opti2.py"

# Define the grid sizes as an array of strings "NX NY"
grids=(
    "80 10"
    "200 100"
    "500 100"
    "1000 100"
    "5000 100"
    "1000 1000"
    "5000 1000"
)

echo "====================================================="
echo " Starting Automated Warp CFD Benchmarks on RTX 6000  "
echo "====================================================="

# Loop through each grid size in the array
for grid in "${grids[@]}"; do
    # Extract NX and NY from the string
    read -r NX NY <<< "$grid"
    
    # Define the log file name
    LOG_FILE="benchmark_log_${NX}x${NY}.txt"
    
    echo "-> Running Grid: ${NX} x ${NY}..."
    echo "-> Logging to: ${LOG_FILE}"
    
    # Execute the Python script. 
    # The '-u' flag forces Python to output unbuffered logs, so you can track progress in real-time.
    # '> "$LOG_FILE" 2>&1' redirects both standard output and errors into the log file.
    python -u $PYTHON_SCRIPT $NX $NY > "$LOG_FILE" 2>&1
    
    echo "-> Finished Grid: ${NX} x ${NY}"
    echo "-----------------------------------------------------"
done

echo "All benchmark runs completed successfully!"
