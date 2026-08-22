#!/bin/bash

# The metrics we want to extract
METRICS="smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed"

# Targeting the first stage of your extreme fused RK3 kernel
KERNEL_REGEX="compute_fluxes_kernel"

# The exact absolute path you just found
NCU_CMD="/usr/local/cuda/bin/ncu" 

for GRID in "80 10" "200 100" "500 100" "1000 100" "5000 100" "1000 1000" "5000 1000"; do
    read NX NY <<< $GRID
    echo "========================================="
    echo " Profiling Grid: ${NX}x${NY}"
    echo "========================================="
    
    $NCU_CMD \
        --metrics $METRICS \
        --set full \
        --kernel-name regex:$KERNEL_REGEX \
        --launch-count 3 \
        python FDM_WARP_test_opti1.py $NX $NY --profile
        
    echo ""
done
