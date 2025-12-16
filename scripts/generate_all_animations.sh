#!/bin/bash

# Generate GIF animations for all scenarios 01-10
echo "Starting batch animation generation..."

for k in {1..10}; do
    i=$(printf "%02d" $k)
    echo "----------------------------------------------------------------"
    echo "Generating animation for scenario_$i..."
    echo "----------------------------------------------------------------"
    python examples/run_simulation.py \
        --scenario scenarios/scenario_$i.yaml \
        --animate \
        --animation-format gif \
        --fps 10 \
        --log-level INFO
done

echo "Batch animation generation complete!"
