"""GreenLake consumption vs Own (CapEx) vs AWS EC2 (OpEx).

A third cost mode: HPE GreenLake bills GPU hardware that sits in your facility as metered
OpEx (committed baseline + burst), so data and model weights can stay on-prem while the
spend behaves like cloud. This page puts that mode side by side with owning the box
outright (CapEx) and renting equivalent GPUs from AWS EC2 (OpEx).
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data import (
    GPU_CATALOG,
    CHASSIS_CATALOG,
    EC2_INSTANCES,
    ONPREM_DEFAULTS,
    GREENLAKE_DEFAULTS,
    DAYS_PER_MONTH,
    calc_onprem_capex,
    calc_onprem_monthly_opex,
    estimate_system_power,
    calc_greenlake_monthly,
)

st.set_page_config(page_title="GreenLake", page_icon=":money_with_wings:", layout="wide")

st.title("GreenLake Consumption vs Own vs EC2", anchor=False)
st.caption(
    "Three ways to get GPU: own it (CapEx), rent AWS EC2 (OpEx), or HPE GreenLake "
    "on-prem consumption (metered OpEx, hardware in your facility)."
)

st.warning(
    "GreenLake pricing is negotiated and not public. The committed-baseline and burst rates "
    "here are **placeholders** (`GREENLAKE_DEFAULTS` in `data.py`). Replace them with an "
    "actual vendor quote before trusting any breakeven."
)

COLORS = {"Own (CapEx)": "#2563EB", "AWS EC2 (OpEx)": "#FF9900", "GreenLake (consumption)": "#01A982"}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=70, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
    hoverlabel=dict(namelength=-1),
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Scenario")

gpu_name = st.sidebar.selectbox("GPU model (owned + GreenLake tiers)", list(GPU_CATALOG.keys()),
                                index=list(GPU_CATALOG.keys()).index("L40S") if "L40S" in GPU_CATALOG else 0)
gpu = GPU_CATALOG[gpu_name]

chassis_name = st.sidebar.selectbox("Chassis (owned)", list(CHASSIS_CATALOG.keys()),
                                    index=list(CHASSIS_CATALOG.keys()).index("HPE DL380a Gen11 (2U, 4 GPU)")
                                    if "HPE DL380a Gen11 (2U, 4 GPU)" in CHASSIS_CATALOG else 0)
chassis = CHASSIS_CATALOG[chassis_name]

# Cap GPU count at the chassis capacity so Own CapEx stays a physically valid single-chassis build.
max_gpus = int(chassis["max_gpus"])
gpu_count = st.sidebar.slider("GPU count", 1, max_gpus, min(4, max_gpus),
                              help="Capped at the selected chassis capacity (one chassis is priced).")

ec2_name = st.sidebar.selectbox("AWS EC2 instance", list(EC2_INSTANCES.keys()))
ec2 = EC2_INSTANCES[ec2_name]
ec2_tier = st.sidebar.radio("EC2 pricing", ["on_demand_hr", "spot_hr", "reserved_1yr_hr"],
                            format_func=lambda k: {"on_demand_hr": "On-Demand", "spot_hr": "Spot",
                                                   "reserved_1yr_hr": "Reserved (1yr)"}[k])

st.sidebar.header("Duty cycle")
hours_per_day = st.sidebar.slider("GPU hours run per day", 1, 24, 8,
                                  help="Drives EC2 hours and GreenLake burst. Low/spiky favors consumption; hot 24/7 favors owning.")

st.sidebar.header("GreenLake (placeholders)")
gl_baseline = st.sidebar.number_input("Committed baseline $/mo (take-or-pay)", min_value=0.0, value=float(GREENLAKE_DEFAULTS["committed_baseline_monthly"]), step=500.0)
gl_committed_hours = st.sidebar.number_input("GPU-hours covered by baseline / mo", min_value=0.0, value=float(GREENLAKE_DEFAULTS["committed_gpu_hours_month"]), step=100.0)
gl_burst = st.sidebar.number_input("Burst $/GPU-hour above baseline", min_value=0.0, value=float(GREENLAKE_DEFAULTS["burst_rate_gpu_hr"]), step=0.25)
gl_facility = st.sidebar.checkbox("Facility power + rack are ours", value=GREENLAKE_DEFAULTS["facility_is_ours"],
                                  help="GreenLake hardware sits in your DC, so power/cooling/space are usually your cost.")

# ---------------------------------------------------------------------------
# Cost math
# ---------------------------------------------------------------------------
hours_per_month = hours_per_day * DAYS_PER_MONTH
power_kw = estimate_system_power(gpu_count, gpu["tdp_w"])
facility_opex = calc_onprem_monthly_opex(power_kw)

# Own (CapEx + power/rack opex, runs 24/7 available)
own_capex = calc_onprem_capex(gpu_count, gpu["price_default"], chassis["price_default"])
own_monthly = facility_opex

# AWS EC2 (OpEx). Normalize the instance's bundled GPUs to the chosen gpu_count.
ec2_per_gpu_hr = ec2[ec2_tier] / max(1, ec2["gpu_count"])
ec2_monthly = ec2_per_gpu_hr * gpu_count * hours_per_month

# GreenLake (on-prem consumption). GPU-hours actually consumed this month.
total_gpu_hours = gpu_count * hours_per_month
gl_monthly = calc_greenlake_monthly(
    total_gpu_hours,
    committed_baseline_monthly=gl_baseline,
    committed_gpu_hours_month=gl_committed_hours,
    burst_rate_gpu_hr=gl_burst,
    facility_opex_monthly=facility_opex if gl_facility else 0.0,
)
gl_install = float(GREENLAKE_DEFAULTS["install_fee"])

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Own (CapEx)", f"${own_capex:,.0f}", help=f"Plus ${own_monthly:,.0f}/mo power + rack")
c2.metric("AWS EC2 (monthly)", f"${ec2_monthly:,.0f}/mo", help=f"{ec2['gpu_count']}x {ec2['gpu_model']} @ {hours_per_day}h/day")
c3.metric("GreenLake (monthly)", f"${gl_monthly:,.0f}/mo", help="On-prem, metered: committed baseline + burst")
st.caption(
    "Read with care: the owned/GreenLake GPU and the EC2 GPU are picked independently, so compare "
    "comparable models. GreenLake's committed baseline is a fixed reservation (the GPU-hours set at "
    "left, roughly one GPU), so at higher GPU counts it reads as 'reserved + burst' versus N "
    "dedicated GPUs and can look cheaper than a like-for-like reservation would."
)

# ---------------------------------------------------------------------------
# 36-month cumulative
# ---------------------------------------------------------------------------
months = list(range(0, 37))
rows = []
for m in months:
    rows.append({
        "month": m,
        "Own (CapEx)": own_capex + own_monthly * m,
        "AWS EC2 (OpEx)": ec2_monthly * m,
        "GreenLake (consumption)": gl_install + gl_monthly * m,
    })
df = pd.DataFrame(rows)

fig = go.Figure()
for series, color in COLORS.items():
    fig.add_trace(go.Scatter(x=df["month"], y=df[series], name=series, mode="lines",
                             line=dict(color=color, width=3)))
fig.update_layout(xaxis_title="Month", yaxis_title="Cumulative cost ($)", **PLOTLY_LAYOUT)
st.subheader("Cumulative cost over 36 months", anchor=False)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Power/scaling note: EC2 scales fully with the duty-cycle slider. For GreenLake only the burst "
    "above the committed baseline scales; the committed baseline and (if facility is ours) on-prem "
    "power are modeled continuous, like Own. So at low duty cycle Own and GreenLake both still carry "
    "full facility power while EC2 does not."
)


def breakeven(capex_a, monthly_a, capex_b, monthly_b):
    """Month where option A (higher upfront) becomes cheaper than option B. None if never."""
    delta = monthly_b - monthly_a
    gap = capex_a - capex_b
    if delta <= 0:
        return None
    m = gap / delta
    return m if m > 0 else 0.0


be_own_vs_ec2 = breakeven(own_capex, own_monthly, 0.0, ec2_monthly)
be_own_vs_gl = breakeven(own_capex, own_monthly, gl_install, gl_monthly)

st.subheader("Break-even", anchor=False)
b1, b2, b3 = st.columns(3)
b1.metric("Own pays off vs EC2", f"{be_own_vs_ec2:.1f} mo" if be_own_vs_ec2 else "never")
b2.metric("Own pays off vs GreenLake", f"{be_own_vs_gl:.1f} mo" if be_own_vs_gl else "never")
# GreenLake has no CapEx (install_fee = 0), so against EC2 it is simply cheaper now or never.
b3.metric("GreenLake beats EC2", "cheaper now" if gl_monthly < ec2_monthly else "never")

# ---------------------------------------------------------------------------
# TCO table
# ---------------------------------------------------------------------------
tco_rows = []
for years in [1, 2, 3]:
    m = years * 12
    costs = {
        "Own (CapEx)": own_capex + own_monthly * m,
        "AWS EC2 (OpEx)": ec2_monthly * m,
        "GreenLake (consumption)": gl_install + gl_monthly * m,
    }
    tco_rows.append({
        "Horizon": f"{years} Year{'s' if years > 1 else ''}",
        **{k: f"${v:,.0f}" for k, v in costs.items()},
        "Cheapest": min(costs, key=costs.get),
    })
st.subheader("Total cost of ownership", anchor=False)
st.dataframe(pd.DataFrame(tco_rows), hide_index=True, use_container_width=True)

st.info(
    "Data-residency note: GreenLake keeps the hardware physically on-prem, so data and model "
    "weights can stay in your facility. The tradeoff is the metering plane, a phone-home that "
    "gates capacity, so if data residency is a hard requirement, confirm with the vendor in "
    "writing that the GPUs keep running through a network disconnection (store-and-forward "
    "metering) rather than throttling or stopping."
)
