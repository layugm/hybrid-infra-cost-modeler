"""Smoke tests for the cost model in data.py.

Kept dependency-light (no Streamlit) so CI can run them headless. These exist so
`pytest` has something to collect; previously it exited 5 (no tests) and failed CI.
"""

import data


def test_onprem_capex_adds_components():
    capex = data.calc_onprem_capex(gpu_count=4, gpu_unit_cost=5000, chassis=10000,
                                   ram=2500, storage=2500)
    assert capex == 10000 + 4 * 5000 + 2500 + 2500


def test_cloud_monthly_scales_with_hours():
    monthly = data.calc_cloud_monthly(hourly_rate=2.0, hours_per_day=10)
    assert monthly == 2.0 * 10 * data.DAYS_PER_MONTH


def test_breakeven_none_when_cloud_is_cheaper():
    assert data.calc_breakeven_months(capex=10000, onprem_monthly=500,
                                      cloud_monthly=300) is None


def test_greenlake_take_or_pay_floor():
    # Usage below the committed baseline still bills the full committed baseline.
    monthly = data.calc_greenlake_monthly(
        total_gpu_hours=100.0,
        committed_baseline_monthly=4000.0,
        committed_gpu_hours_month=730.0,
        burst_rate_gpu_hr=3.5,
        facility_opex_monthly=0.0,
    )
    assert monthly == 4000.0


def test_greenlake_burst_above_baseline():
    # 100 GPU-hours above the 730 baseline, billed at the burst rate, plus facility opex.
    monthly = data.calc_greenlake_monthly(
        total_gpu_hours=830.0,
        committed_baseline_monthly=4000.0,
        committed_gpu_hours_month=730.0,
        burst_rate_gpu_hr=3.5,
        facility_opex_monthly=200.0,
    )
    assert monthly == 4000.0 + 100.0 * 3.5 + 200.0
