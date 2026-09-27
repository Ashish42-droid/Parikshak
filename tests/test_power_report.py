"""Board power from tegrastats, checked on real line formats from both Jetson generations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import power_report  # noqa: E402

ORIN = ("09-15-2026 10:00:01 RAM 2140/7620MB (lfb 2x4MB) CPU [12%@1510,8%@1510] "
        "GR3D_FREQ 43% cpu@48.5C gpu@47.1C VDD_IN {w}mW/5316mW VDD_CPU_GPU_CV 1203mW/1203mW "
        "VDD_SOC 1602mW/1602mW")
NANO = ("RAM 1890/3956MB (lfb 120x4MB) CPU [30%@1479,20%@1479] GR3D_FREQ 61%@921 "
        "POM_5V_IN {w}/3405 POM_5V_GPU 982/982 POM_5V_CPU 804/804")


def test_the_orin_input_rail_is_read_and_the_sub_rails_are_not_added():
    watts = power_report.parse([ORIN.format(w=5316), ORIN.format(w=7400)])
    assert watts == [5.316, 7.4]


def test_the_older_input_rail_format_is_read():
    assert power_report.parse([NANO.format(w=3405)]) == [3.405]


def test_the_gate_needs_both_halves():
    watts = [9.0] * 50 + [26.0]
    over = power_report.summarise(watts, fps=14.0)
    assert not over["power_ok"] and over["verdict"] == "FAIL" and over["peak_w"] == 26.0
    under = power_report.summarise([9.0] * 50, fps=14.0)
    assert under["verdict"] == "PASS" and under["frames_per_watt"] == round(14.0 / 9.0, 2)
    slow = power_report.summarise([9.0] * 50, fps=8.0)
    assert slow["verdict"] == "FAIL"


def test_a_log_without_power_fields_says_so():
    assert power_report.summarise(power_report.parse(["no rails here"]))["verdict"] == "NO DATA"
