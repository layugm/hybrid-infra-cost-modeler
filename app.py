"""Hybrid Infrastructure Cost Modeler — On-Prem vs Cloud GPU Comparison."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data import (
    GPU_CATALOG,
    CHASSIS_CATALOG,
    ONPREM_DEFAULTS,
    EC2_INSTANCES,
    AWS_REGIONS,
    PRICING_TIERS,
    calc_onprem_capex,
    calc_onprem_monthly_opex,
    calc_cloud_monthly,
    calc_breakeven_months,
    build_timeline_df,
    build_tco_table,
    estimate_system_power,
    fetch_all_live_prices,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Infra Cost Modeler",
    page_icon=":bar_chart:",
    layout="wide",
)

COLORS = {
    "onprem": "#1f77b4",
    "onprem_light": "#aec7e8",
    "cloud": "#ff7f0e",
    "cloud_light": "#ffbb78",
    "hybrid": "#2ca02c",
    "hybrid_light": "#98df8a",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hoverlabel=dict(bgcolor="white"),
)

PLOTLY_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": "cost_comparison",
        "height": 600,
        "width": 1000,
        "scale": 2,
    },
    "displayModeBar": True,
}

# ---------------------------------------------------------------------------
# Sidebar — On-Prem
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")

with st.sidebar.expander("On-Prem Hardware", expanded=True):
    # Chassis
    chassis_name = st.selectbox("Server Chassis", list(CHASSIS_CATALOG.keys()))
    chassis_info = CHASSIS_CATALOG[chassis_name]
    chassis_cost = st.number_input(
        "Chassis Cost ($)",
        min_value=chassis_info["price_low"],
        max_value=chassis_info["price_high"],
        value=chassis_info["price_default"],
        step=500,
    )

    # GPU
    gpu_model = st.selectbox("GPU Model", list(GPU_CATALOG.keys()))
    gpu_info = GPU_CATALOG[gpu_model]
    gpu_count = st.slider("GPU Count", 1, chassis_info["max_gpus"], min(ONPREM_DEFAULTS["gpu_count"], chassis_info["max_gpus"]))
    gpu_unit_cost = st.number_input(
        "GPU Unit Cost ($)",
        min_value=gpu_info["price_low"],
        max_value=gpu_info["price_high"],
        value=gpu_info["price_default"],
        step=100,
    )

    # Other hardware
    ram_cost = st.number_input("RAM Cost ($)", value=ONPREM_DEFAULTS["ram_cost"], step=100)
    storage_cost = st.number_input("Storage/PSU/Rails ($)", value=ONPREM_DEFAULTS["storage_cost"], step=100)

    # Power
    estimated_power = estimate_system_power(gpu_count, gpu_info["tdp_w"])
    power_kw = st.number_input(
        "System Power (kW)",
        value=round(estimated_power, 1),
        step=0.1,
        help=f"Auto-estimated from {gpu_count}x {gpu_info['tdp_w']}W GPUs + 500W base. Override if you know your actual draw.",
    )
    electricity = st.number_input(
        "Electricity ($/kWh)",
        value=ONPREM_DEFAULTS["electricity_kwh"],
        step=0.01,
        format="%.2f",
    )
    rack_monthly = st.number_input("Monthly Rack Cost ($)", value=ONPREM_DEFAULTS["rack_monthly"], step=50)

# ---------------------------------------------------------------------------
# Sidebar — Cloud
# ---------------------------------------------------------------------------
with st.sidebar.expander("Cloud (AWS EC2)", expanded=True):
    aws_region = st.selectbox("AWS Region", list(AWS_REGIONS.keys()),
                              format_func=lambda r: f"{r} — {AWS_REGIONS[r]}")

    # Live pricing toggle
    use_live = st.toggle("Live AWS pricing", value=False,
                         help="Fetch real-time spot and on-demand prices via boto3. Requires AWS credentials configured.")

    live_prices = None
    if use_live:
        cache_key = f"live_prices_{aws_region}"
        if cache_key not in st.session_state:
            with st.spinner("Fetching live AWS prices..."):
                st.session_state[cache_key] = fetch_all_live_prices(aws_region)
        live_prices = st.session_state[cache_key]

        fetched_at = live_prices.get("fetched_at", "")
        if fetched_at:
            st.caption(f"Fetched: {fetched_at[:19]}Z")
        if st.button("Refresh prices"):
            with st.spinner("Refreshing..."):
                st.session_state[cache_key] = fetch_all_live_prices(aws_region)
                st.rerun()

    # Instance selection with summary
    def format_instance(name):
        inst = EC2_INSTANCES[name]
        return f"{name} — {inst['gpu_count']}x {inst['gpu_model']} ({inst['vram_gb']}GB)"

    instance_name = st.selectbox("Instance Type", list(EC2_INSTANCES.keys()), format_func=format_instance)
    instance = EC2_INSTANCES[instance_name]

    # Show instance specs
    st.caption(f"{instance['vcpus']} vCPUs · {instance['ram_gb']} GiB RAM · {instance['vram_gb']} GB VRAM")

    pricing_tier = st.radio("Pricing Tier", list(PRICING_TIERS.keys()))

    # Determine hourly rate
    static_rate = instance[PRICING_TIERS[pricing_tier]]
    hourly_rate = static_rate

    if live_prices:
        live_spot = live_prices["spot"].get(instance_name)
        live_od = live_prices["on_demand"].get(instance_name)

        if pricing_tier == "Spot" and live_spot is not None:
            hourly_rate = live_spot
            delta = live_spot - instance["spot_hr"]
            st.metric("Live Spot Rate", f"${live_spot:.4f}/hr",
                      delta=f"${delta:+.4f} vs static", delta_color="inverse")
        elif pricing_tier == "On-Demand" and live_od is not None:
            hourly_rate = live_od
            delta = live_od - instance["on_demand_hr"]
            st.metric("Live On-Demand Rate", f"${live_od:.4f}/hr",
                      delta=f"${delta:+.4f} vs static", delta_color="inverse")
        elif pricing_tier == "1-Year Reserved":
            st.caption("Reserved pricing not available via API — using static rate.")
    else:
        st.caption(f"Rate: ${static_rate:.2f}/hr (static)")

    usage_pattern = st.selectbox("Usage Pattern", ["24/7", "Business hours (10h weekdays)", "Custom"])
    if usage_pattern == "24/7":
        hours_per_day = 24.0
    elif usage_pattern.startswith("Business"):
        hours_per_day = 10.0 * (5.0 / 7.0)
    else:
        hours_per_day = st.slider("Hours per day", 1.0, 24.0, 12.0, 0.5)

# ---------------------------------------------------------------------------
# Sidebar — Analysis
# ---------------------------------------------------------------------------
with st.sidebar.expander("Analysis Settings", expanded=False):
    horizon_months = st.slider("Analysis Horizon (months)", 12, 60, 36)
    show_hybrid = st.checkbox("Include hybrid scenario", value=True,
                              help="Hybrid = on-prem hardware costs + cloud compute costs combined.")

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
capex = calc_onprem_capex(gpu_count, gpu_unit_cost, chassis_cost, ram_cost, storage_cost)
onprem_monthly = calc_onprem_monthly_opex(power_kw, electricity, rack_monthly)
cloud_monthly = calc_cloud_monthly(hourly_rate, hours_per_day)
breakeven = calc_breakeven_months(capex, onprem_monthly, cloud_monthly)
hybrid_monthly = onprem_monthly + cloud_monthly if show_hybrid else None

# ---------------------------------------------------------------------------
# Title + context
# ---------------------------------------------------------------------------
st.title("Hybrid Infrastructure Cost Modeler")
st.caption("Compare on-prem GPU servers against AWS EC2 — interactive CapEx vs OpEx analysis with live pricing support.")

# ---------------------------------------------------------------------------
# Section A: Summary metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("On-Prem CapEx", f"${capex:,.0f}")
c2.metric("On-Prem Monthly OpEx", f"${onprem_monthly:,.0f}")
c3.metric("Cloud Monthly", f"${cloud_monthly:,.0f}")
if breakeven is not None:
    c4.metric("Break-Even", f"{breakeven:.1f} months")
else:
    c4.metric("Break-Even", "Never (cloud cheaper)")

# Config summary
with st.expander("Current configuration summary"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""**On-Prem**
- {chassis_name}
- {gpu_count}x {gpu_model} ({gpu_count * gpu_info['vram_gb']} GB total VRAM)
- ~{power_kw:.1f} kW draw @ ${electricity:.2f}/kWh
- CapEx: ${capex:,.0f} | OpEx: ${onprem_monthly:,.0f}/mo""")
    with col2:
        st.markdown(f"""**Cloud**
- {instance_name} ({instance['gpu_count']}x {instance['gpu_model']})
- {instance['vram_gb']} GB VRAM | {instance['vcpus']} vCPUs | {instance['ram_gb']} GiB RAM
- {pricing_tier}: ${hourly_rate:.2f}/hr {'(LIVE)' if use_live and live_prices else '(static)'}
- {usage_pattern}: ${cloud_monthly:,.0f}/mo""")

st.divider()

# ---------------------------------------------------------------------------
# Section B: Break-even timeline
# ---------------------------------------------------------------------------
st.subheader("Cumulative Cost Over Time")

timeline = build_timeline_df(capex, onprem_monthly, cloud_monthly, horizon_months, hybrid_monthly)

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(
    x=timeline["month"], y=timeline["onprem_cumulative"],
    name="On-Prem", line=dict(color=COLORS["onprem"], width=3),
    hovertemplate="Month %{x}<br>$%{y:,.0f}<extra>On-Prem</extra>",
))
fig_timeline.add_trace(go.Scatter(
    x=timeline["month"], y=timeline["cloud_cumulative"],
    name="Cloud", line=dict(color=COLORS["cloud"], width=3),
    hovertemplate="Month %{x}<br>$%{y:,.0f}<extra>Cloud</extra>",
))
if show_hybrid and "hybrid_cumulative" in timeline.columns:
    fig_timeline.add_trace(go.Scatter(
        x=timeline["month"], y=timeline["hybrid_cumulative"],
        name="Hybrid (on-prem + cloud)",
        line=dict(color=COLORS["hybrid"], width=3, dash="dash"),
        hovertemplate="Month %{x}<br>$%{y:,.0f}<extra>Hybrid</extra>",
    ))

if breakeven is not None and breakeven <= horizon_months:
    fig_timeline.add_vline(
        x=breakeven, line_dash="dot", line_color="gray",
        annotation_text=f"Break-even: {breakeven:.1f}mo",
        annotation_position="top right",
    )

fig_timeline.update_layout(
    **PLOTLY_LAYOUT,
    xaxis_title="Months",
    yaxis_title="Cumulative Cost ($)",
    yaxis_tickformat="$,.0f",
)
st.plotly_chart(fig_timeline, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------------------
# Section C: Monthly cost breakdown
# ---------------------------------------------------------------------------
st.subheader("Monthly Cost Breakdown")

amort_months = max(horizon_months, 1)
onprem_amortized = capex / amort_months

categories = ["On-Prem", "Cloud"]
amort_values = [onprem_amortized, 0]
opex_values = [onprem_monthly, cloud_monthly]

if show_hybrid:
    categories.append("Hybrid")
    amort_values.append(onprem_amortized)
    opex_values.append(onprem_monthly + cloud_monthly)

fig_monthly = go.Figure()
fig_monthly.add_trace(go.Bar(
    x=categories, y=amort_values,
    name="CapEx Amortization",
    marker_color=COLORS["onprem_light"],
    hovertemplate="%{x}<br>$%{y:,.0f}/mo<extra>CapEx Amort.</extra>",
))
fig_monthly.add_trace(go.Bar(
    x=categories, y=opex_values,
    name="Monthly OpEx",
    marker_color=[COLORS["onprem"], COLORS["cloud"], COLORS["hybrid"]][:len(categories)],
    hovertemplate="%{x}<br>$%{y:,.0f}/mo<extra>OpEx</extra>",
))
fig_monthly.update_layout(
    **PLOTLY_LAYOUT,
    barmode="stack",
    yaxis_title="Monthly Cost ($)",
    yaxis_tickformat="$,.0f",
)
st.plotly_chart(fig_monthly, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------------------
# Section D: VRAM comparison
# ---------------------------------------------------------------------------
st.subheader("VRAM Comparison")

vram_labels = [f"On-Prem ({gpu_count}x {gpu_model})"]
vram_values = [gpu_count * gpu_info["vram_gb"]]
vram_colors = [COLORS["onprem"]]

for name, inst in EC2_INSTANCES.items():
    vram_labels.append(f"{name} ({inst['gpu_count']}x {inst['gpu_model']})")
    vram_values.append(inst["vram_gb"])
    vram_colors.append(COLORS["cloud"] if name == instance_name else COLORS["cloud_light"])

fig_vram = go.Figure(go.Bar(
    y=vram_labels,
    x=vram_values,
    orientation="h",
    marker_color=vram_colors,
    text=[f"{v} GB" for v in vram_values],
    textposition="outside",
    hovertemplate="%{y}<br>%{x} GB<extra></extra>",
))
fig_vram.update_layout(
    **PLOTLY_LAYOUT,
    xaxis_title="Total VRAM (GB)",
    height=max(300, len(vram_labels) * 35),
)
st.plotly_chart(fig_vram, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------------------
# Section E: TCO table + chart
# ---------------------------------------------------------------------------
st.subheader("Total Cost of Ownership")

tco = build_tco_table(capex, onprem_monthly, cloud_monthly, hybrid_monthly)

tco_display = tco.copy()
for col in tco_display.columns:
    if "TCO" in col:
        tco_display[col] = tco_display[col].apply(lambda v: f"${v:,.0f}")

st.dataframe(tco_display, use_container_width=True, hide_index=True)

fig_tco = go.Figure()
fig_tco.add_trace(go.Bar(
    x=tco["Horizon"], y=tco["On-Prem TCO"],
    name="On-Prem", marker_color=COLORS["onprem"],
    hovertemplate="%{x}<br>$%{y:,.0f}<extra>On-Prem</extra>",
))
fig_tco.add_trace(go.Bar(
    x=tco["Horizon"], y=tco["Cloud TCO"],
    name="Cloud", marker_color=COLORS["cloud"],
    hovertemplate="%{x}<br>$%{y:,.0f}<extra>Cloud</extra>",
))
if show_hybrid and "Hybrid TCO" in tco.columns:
    fig_tco.add_trace(go.Bar(
        x=tco["Horizon"], y=tco["Hybrid TCO"],
        name="Hybrid", marker_color=COLORS["hybrid"],
        hovertemplate="%{x}<br>$%{y:,.0f}<extra>Hybrid</extra>",
    ))
fig_tco.update_layout(
    **PLOTLY_LAYOUT,
    barmode="group",
    yaxis_title="Total Cost ($)",
    yaxis_tickformat="$,.0f",
)
st.plotly_chart(fig_tco, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------------------
# Section F: Export
# ---------------------------------------------------------------------------
st.subheader("Export")

csv = timeline.to_csv(index=False)
st.download_button(
    label="Download timeline data (CSV)",
    data=csv,
    file_name="cost_timeline.csv",
    mime="text/csv",
)
st.caption("Use the camera icon in the top-right of each chart to export as PNG.")
