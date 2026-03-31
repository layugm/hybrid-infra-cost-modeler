"""Advanced Analysis — Utilization, Scenarios, Executive Summary."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data import (
    EC2_INSTANCES,
    PRICING_TIERS,
    GLOSSARY,
    calc_onprem_capex,
    calc_onprem_monthly_opex,
    calc_fleet_monthly,
    calc_fleet_vram,
    calc_breakeven_months,
    build_timeline_df,
    build_tco_table,
    apply_utilization,
    generate_summary,
)

st.set_page_config(page_title="Advanced Analysis", page_icon=":mag:", layout="wide")

COLORS = {
    "onprem": "#1f77b4",
    "onprem_light": "#aec7e8",
    "cloud": "#ff7f0e",
    "hybrid": "#2ca02c",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hoverlabel=dict(bgcolor="white"),
)

PLOTLY_CONFIG = {"displayModeBar": True}

# ---------------------------------------------------------------------------
# Check for computed data from main page
# ---------------------------------------------------------------------------
if "computed" not in st.session_state:
    st.warning("Configure your infrastructure on the main **Cost Modeler** page first, then return here.")
    st.stop()

c = st.session_state["computed"]

st.title("Advanced Analysis")
st.caption("Deeper analysis tools — utilization modeling, scenario comparison, and executive summary.")

# ---------------------------------------------------------------------------
# Section 1: Utilization Modeling
# ---------------------------------------------------------------------------
st.header("Utilization Modeling")
st.caption(
    "On-prem GPUs aren't always 100% busy. Lower utilization means you're paying "
    "for idle capacity, increasing your effective cost per unit of work."
)

utilization = st.slider(
    "On-prem GPU utilization (%)", 10, 100, 70, 5,
    help="What percentage of time your GPUs are actively doing useful work. "
         "70% is typical for shared clusters. 40-50% is common for dev/test.",
)

effective_capex = apply_utilization(c["capex"], utilization)
adjusted_onprem_monthly = c["onprem_monthly"]  # OpEx doesn't change with utilization
adjusted_breakeven = calc_breakeven_months(effective_capex, adjusted_onprem_monthly, c["cloud_monthly"])

col1, col2, col3 = st.columns(3)
col1.metric("Actual CapEx", f"${c['capex']:,.0f}")
col2.metric("Effective CapEx", f"${effective_capex:,.0f}",
            delta=f"+${effective_capex - c['capex']:,.0f}" if utilization < 100 else None,
            delta_color="inverse",
            help="What your hardware effectively costs when accounting for idle time.")
if adjusted_breakeven:
    col3.metric("Adjusted Break-Even", f"{adjusted_breakeven:.1f} months",
                delta=f"+{adjusted_breakeven - c['breakeven']:.1f}mo" if c["breakeven"] else None,
                delta_color="inverse")
else:
    col3.metric("Adjusted Break-Even", "Never")

# Utilization-adjusted timeline chart
horizon = c["horizon_months"]
timeline_actual = build_timeline_df(c["capex"], c["onprem_monthly"], c["cloud_monthly"], horizon)
timeline_adjusted = build_timeline_df(effective_capex, adjusted_onprem_monthly, c["cloud_monthly"], horizon)

fig_util = go.Figure()
fig_util.add_trace(go.Scatter(
    x=timeline_actual["month"], y=timeline_actual["onprem_cumulative"],
    name="On-Prem (100% utilization)", line=dict(color=COLORS["onprem_light"], width=2, dash="dot"),
    hovertemplate="Month %{x}<br>$%{y:,.0f}<extra>100% util</extra>",
))
fig_util.add_trace(go.Scatter(
    x=timeline_adjusted["month"], y=timeline_adjusted["onprem_cumulative"],
    name=f"On-Prem ({utilization}% utilization)", line=dict(color=COLORS["onprem"], width=3),
    hovertemplate="Month %{x}<br>$%{y:,.0f}<extra>" + f"{utilization}% util</extra>",
))
fig_util.add_trace(go.Scatter(
    x=timeline_actual["month"], y=timeline_actual["cloud_cumulative"],
    name="Cloud Fleet", line=dict(color=COLORS["cloud"], width=3),
    hovertemplate="Month %{x}<br>$%{y:,.0f}<extra>Cloud</extra>",
))
if adjusted_breakeven and adjusted_breakeven <= horizon:
    fig_util.add_vline(x=adjusted_breakeven, line_dash="dot", line_color="gray",
                       annotation_text=f"Break-even: {adjusted_breakeven:.1f}mo")
fig_util.update_layout(**PLOTLY_LAYOUT, xaxis_title="Months",
                       yaxis_title="Cumulative Cost ($)", yaxis_tickformat="$,.0f")
st.plotly_chart(fig_util, use_container_width=True, config=PLOTLY_CONFIG)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Scenario Snapshots
# ---------------------------------------------------------------------------
st.header("Scenario Comparison")
st.caption("Save different configurations and compare them side-by-side.")

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}

# Save current scenario
with st.expander("Save / Manage Scenarios"):
    save_col1, save_col2 = st.columns([3, 1])
    scenario_name = save_col1.text_input("Scenario name", placeholder="e.g., Conservative, Cloud-heavy, Hybrid")
    if save_col2.button("Save Current") and scenario_name:
        st.session_state["scenarios"][scenario_name] = {
            "capex": c["capex"],
            "onprem_monthly": c["onprem_monthly"],
            "cloud_monthly": c["cloud_monthly"],
            "fleet_monthly": c["fleet_monthly"],
            "eks_monthly": c["eks_monthly"],
            "dx_monthly": c["dx_monthly"],
            "breakeven": c["breakeven"],
            "gpu_model": c["gpu_model"],
            "gpu_count": c["gpu_count"],
            "chassis_name": c["chassis_name"],
            "fleet_entries": c["fleet_entries"],
            "fleet_vram": calc_fleet_vram(c["fleet_entries"]),
            "onprem_vram": c["gpu_count"] * c["gpu_info"]["vram_gb"],
        }
        st.success(f"Saved: {scenario_name}")
        st.rerun()

    # Delete scenarios
    if st.session_state["scenarios"]:
        del_name = st.selectbox("Delete a scenario", [""] + list(st.session_state["scenarios"].keys()))
        if del_name and st.button(f"Delete '{del_name}'"):
            del st.session_state["scenarios"][del_name]
            st.rerun()

# Compare scenarios
scenarios = st.session_state["scenarios"]
if len(scenarios) >= 2:
    selected = st.multiselect("Select scenarios to compare", list(scenarios.keys()),
                              default=list(scenarios.keys())[:3])

    if len(selected) >= 2:
        # Comparison table
        rows = []
        for name in selected:
            s = scenarios[name]
            rows.append({
                "Scenario": name,
                "CapEx": f"${s['capex']:,.0f}",
                "On-Prem/mo": f"${s['onprem_monthly']:,.0f}",
                "Cloud/mo": f"${s['cloud_monthly']:,.0f}",
                "Break-Even": f"{s['breakeven']:.1f}mo" if s["breakeven"] else "Never",
                "On-Prem VRAM": f"{s['onprem_vram']} GB",
                "Fleet VRAM": f"{s['fleet_vram']} GB",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # TCO comparison chart
        fig_compare = go.Figure()
        for name in selected:
            s = scenarios[name]
            tco_vals = []
            for years in [1, 2, 3]:
                m = years * 12
                # Use on-prem TCO as the comparison point
                tco_vals.append(s["capex"] + s["onprem_monthly"] * m)
            fig_compare.add_trace(go.Bar(
                x=["1 Year", "2 Years", "3 Years"], y=tco_vals, name=f"{name} (on-prem)",
                hovertemplate="%{x}<br>$%{y:,.0f}<extra>" + name + "</extra>",
            ))

        for name in selected:
            s = scenarios[name]
            cloud_tco = [s["cloud_monthly"] * m for m in [12, 24, 36]]
            fig_compare.add_trace(go.Bar(
                x=["1 Year", "2 Years", "3 Years"], y=cloud_tco, name=f"{name} (cloud)",
                hovertemplate="%{x}<br>$%{y:,.0f}<extra>" + name + " cloud</extra>",
            ))

        fig_compare.update_layout(**PLOTLY_LAYOUT, barmode="group",
                                  yaxis_title="Total Cost ($)", yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_compare, use_container_width=True, config=PLOTLY_CONFIG)
elif len(scenarios) == 1:
    st.info("Save at least 2 scenarios to compare them.")
else:
    st.info("No scenarios saved yet. Configure your infrastructure on the main page, then save it here.")

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Executive Summary
# ---------------------------------------------------------------------------
st.header("Executive Summary")
st.caption("Auto-generated text ready to copy into emails or presentations.")

summary = generate_summary(
    capex=c["capex"],
    onprem_monthly=c["onprem_monthly"],
    cloud_monthly=c["cloud_monthly"],
    breakeven=c["breakeven"],
    gpu_model=c["gpu_model"],
    gpu_count=c["gpu_count"],
    chassis_name=c["chassis_name"],
    fleet=c["fleet_entries"],
    horizon_months=c["horizon_months"],
    eks_monthly=c["eks_monthly"],
    dx_monthly=c["dx_monthly"],
)

st.markdown(f"> {summary}")

st.download_button(
    "Copy summary as text",
    data=summary.replace("**", ""),
    file_name="executive_summary.txt",
    mime="text/plain",
)
