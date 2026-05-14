#!/usr/bin/env bash
set -euo pipefail

windows=(8 16 32)
temps=(0.8 1.0 1.2)

for w in "${windows[@]}"; do
    for t in "${temps[@]}"; do
        echo "=== window=$w temp=$t ==="
        python3 -m src.generation.generate_solo --window "$w" --temperature "$t" --out-dir outputs/solos_sweep
    done
done

python3 -m src.analysis.enrich_json --solos-dir outputs/solos_sweep
