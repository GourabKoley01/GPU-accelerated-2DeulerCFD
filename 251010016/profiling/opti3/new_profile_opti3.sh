#!/bin/bash

# The metrics we want to extract
METRICS="smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"

# Targeting the first stage of your extreme fused RK3 kernel
KERNEL_REGEX="compute_fluxes_kernel"

# The exact absolute path you just found
NCU_CMD="/usr/local/cuda/bin/ncu" 

for GRID in "128 16" "256 32" "512 64" "1024 128" "2048 256" "4096 512" "8192 1024" "16384 2048"; do
    read NX NY <<< $GRID
    echo "========================================="
    echo " Profiling Grid: ${NX}x${NY}"
    echo "========================================="
    
    $NCU_CMD \
        --metrics $METRICS \
        --set full \
        --kernel-name regex:$KERNEL_REGEX \
        --launch-count 1 \
        python FDM_WARP_test_opti3.py $NX $NY --profile
        
    echo ""
done
