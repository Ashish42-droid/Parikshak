#!/usr/bin/env bash
# Log Jetson board power while a benchmark runs. NOT YET RUN ON HARDWARE.
#
#   ./deploy/jetson/log_power.sh runs/power/tegrastats.log &
#   python tools/bench.py --camera 0 --seconds 60
#   kill %1
#   python tools/power_report.py runs/power/tegrastats.log --bench runs/bench.json
set -euo pipefail

out="${1:-runs/power/tegrastats.log}"
mkdir -p "$(dirname "$out")"
echo "nvpmodel: $(nvpmodel -q 2>/dev/null | tr '\n' ' ')" > "${out%.log}.mode.txt"
exec tegrastats --interval 200 --logfile "$out"
