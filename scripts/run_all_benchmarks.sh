#!/bin/bash

# Run benchmarks for all scenarios
echo "Starting batch benchmark..."

for k in {1..3}; do
    i=$(printf "%02d" $k)
    echo "----------------------------------------------------------------"
    echo "Running benchmark for scenario_$i..."
    echo "----------------------------------------------------------------"
    python examples/benchmark_prediction.py --scenario scenarios/scenario_$i.yaml
done

echo "Batch benchmark complete!"
