"""Hybrid Infrastructure Cost Modeler — On-Prem vs Cloud GPU Fleet Comparison."""

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
    calc_fleet_monthly,
    calc_fleet_vram,
    calc_eks_total_monthly,
    apply_secure_workspace,
    build_timeline_df,
    build_tco_table,
    estimate_system_power,
    fetch_all_live_prices,
    calc_direct_connect_monthly,
    GLOSSARY,
    DIRECT_CONNECT_PRICING,
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
    hoverlabel=dict(namelength=-1),
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


def format_instance(name):
    inst = EC2_INSTANCES[name]
    return f"{name} — {inst['gpu_count']}x {inst['gpu_model']} ({inst['vram_gb']}GB)"


# ---------------------------------------------------------------------------
# Initialize fleet in session state
# ---------------------------------------------------------------------------
if "fleet" not in st.session_state:
    st.session_state["fleet"] = [
        {"id": 0, "instance_type": "g6e.12xlarge", "count": 1,
         "pricing_tier": "Spot", "usage_pattern": "24/7", "custom_hours": 12.0}
    ]
if "fleet_counter" not in st.session_state:
    st.session_state["fleet_counter"] = 1


# ---------------------------------------------------------------------------
# Sidebar — On-Prem
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")

with st.sidebar.expander("On-Prem Hardware", expanded=True):
    chassis_name = st.selectbox("Server Chassis", list(CHASSIS_CATALOG.keys()))
    chassis_info = CHASSIS_CATALOG[chassis_name]
    chassis_cost = st.number_input(
        "Chassis Cost ($)",
        min_value=chassis_info["price_low"],
        max_value=chassis_info["price_high"],
        value=chassis_info["price_default"],
        step=500,
    )
    gpu_model = st.selectbox("GPU Model", list(GPU_CATALOG.keys()))
    gpu_info = GPU_CATALOG[gpu_model]
    gpu_count = st.slider("GPU Count", 1, chassis_info["max_gpus"],
                          min(ONPREM_DEFAULTS["gpu_count"], chassis_info["max_gpus"]))
    gpu_unit_cost = st.number_input(
        "GPU Unit Cost ($)",
        min_value=gpu_info["price_low"],
        max_value=gpu_info["price_high"],
        value=gpu_info["price_default"],
        step=100,
    )
    ram_cost = st.number_input("RAM Cost ($)", value=ONPREM_DEFAULTS["ram_cost"], step=100)
    storage_cost = st.number_input("Storage/PSU/Rails ($)", value=ONPREM_DEFAULTS["storage_cost"], step=100)

    estimated_power = estimate_system_power(gpu_count, gpu_info["tdp_w"])
    power_kw = st.number_input(
        "System Power (kW)", value=round(estimated_power, 1), step=0.1,
        help=f"Auto-estimated from {gpu_count}x {gpu_info['tdp_w']}W GPUs + 500W base.",
    )
    electricity = st.number_input("Electricity ($/kWh)", value=ONPREM_DEFAULTS["electricity_kwh"],
                                  step=0.01, format="%.2f")
    rack_monthly = st.number_input("Monthly Rack Cost ($)", value=ONPREM_DEFAULTS["rack_monthly"], step=50)

# ---------------------------------------------------------------------------
# Sidebar — Cloud Fleet
# ---------------------------------------------------------------------------
with st.sidebar.expander("Cloud Fleet (AWS EC2)", expanded=True):
    aws_region = st.selectbox("AWS Region", list(AWS_REGIONS.keys()),
                              format_func=lambda r: f"{r} — {AWS_REGIONS[r]}")

    use_live = st.toggle("Live AWS pricing", value=False,
                         help="Fetch real-time spot/on-demand prices via boto3.")
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

    st.markdown("---")

    # Fleet entries
    for i, entry in enumerate(st.session_state["fleet"]):
        eid = entry["id"]
        st.markdown(f"**Instance {i + 1}**")

        entry["instance_type"] = st.selectbox(
            "Type", list(EC2_INSTANCES.keys()), format_func=format_instance,
            index=list(EC2_INSTANCES.keys()).index(entry["instance_type"]),
            key=f"fleet_type_{eid}",
        )
        inst = EC2_INSTANCES[entry["instance_type"]]
        st.caption(f"{inst['vcpus']} vCPUs · {inst['ram_gb']} GiB RAM · {inst['vram_gb']} GB VRAM")
        st.caption(f"_{inst.get('use_case', '')}_  —  {inst.get('desc', '')}")

        c1, c2 = st.columns(2)
        entry["count"] = c1.number_input("Count", 1, 20, entry["count"], key=f"fleet_count_{eid}")

        # Map pricing tier to glossary key for help text
        _tier_help = {"On-Demand": GLOSSARY["on_demand"], "Spot": GLOSSARY["spot"],
                      "1-Year Reserved": GLOSSARY["reserved_1yr"]}
        entry["pricing_tier"] = c2.selectbox(
            "Pricing", list(PRICING_TIERS.keys()),
            index=list(PRICING_TIERS.keys()).index(entry["pricing_tier"]),
            key=f"fleet_tier_{eid}",
            help=_tier_help.get(entry["pricing_tier"], ""),
        )

        entry["usage_pattern"] = st.selectbox(
            "Usage", ["24/7", "Business hours (10h weekdays)", "Custom"],
            index=["24/7", "Business hours (10h weekdays)", "Custom"].index(entry["usage_pattern"]),
            key=f"fleet_usage_{eid}",
        )
        if entry["usage_pattern"] == "Custom":
            entry["custom_hours"] = st.slider("Hours/day", 1.0, 24.0, entry.get("custom_hours", 12.0),
                                              0.5, key=f"fleet_hours_{eid}")

        if len(st.session_state["fleet"]) > 1:
            if st.button("Remove", key=f"fleet_remove_{eid}"):
                st.session_state["fleet"] = [e for e in st.session_state["fleet"] if e["id"] != eid]
                st.rerun()

        if i < len(st.session_state["fleet"]) - 1:
            st.markdown("---")

    if st.button("+ Add instance type"):
        new_id = st.session_state["fleet_counter"]
        st.session_state["fleet_counter"] = new_id + 1
        st.session_state["fleet"].append(
            {"id": new_id, "instance_type": "g6e.xlarge", "count": 1,
             "pricing_tier": "On-Demand", "usage_pattern": "24/7", "custom_hours": 12.0}
        )
        st.rerun()

# ---------------------------------------------------------------------------
# Sidebar — EKS Management
# ---------------------------------------------------------------------------
with st.sidebar.expander("EKS Management (optional)", expanded=False):
    enable_eks = st.checkbox("Add EKS costs", help=GLOSSARY["eks"])
    eks_monthly = 0.0
    if enable_eks:
        eks_clusters = st.number_input("Cluster count", 1, 10, 1)
        eks_extended = st.checkbox("Extended support ($0.60/hr)")
        eks_mode = st.radio("Mode", ["Standard (control plane only)", "Hybrid Nodes", "EKS Anywhere"],
                           help=f"**Standard**: {GLOSSARY['eks']}\n\n**Hybrid Nodes**: {GLOSSARY['eks_hybrid']}\n\n**EKS Anywhere**: {GLOSSARY['eks_anywhere']}")

        eks_hybrid_vcpus = 0
        eks_anywhere = False
        eks_anywhere_term = 1

        if eks_mode == "Hybrid Nodes":
            eks_hybrid_vcpus = st.number_input(
                "On-prem vCPUs registered", 1, 10000, 96,
                help="vCPUs on your on-prem workers managed by EKS. Priced at $0.020/vCPU/hr (tiered).",
            )
        elif eks_mode == "EKS Anywhere":
            eks_anywhere = True
            eks_anywhere_term = 3 if st.radio(
                "Term", ["1 Year ($2,000/mo)", "3 Year ($1,500/mo)"]
            ).startswith("3") else 1

        eks_monthly = calc_eks_total_monthly(
            cluster_count=eks_clusters,
            extended_support=eks_extended,
            hybrid_vcpus=eks_hybrid_vcpus,
            anywhere=eks_anywhere,
            anywhere_term=eks_anywhere_term,
        )
        st.metric("EKS Monthly", f"${eks_monthly:,.0f}")

# ---------------------------------------------------------------------------
# Sidebar — Secure Workspaces
# ---------------------------------------------------------------------------
with st.sidebar.expander("Secure Workspaces (optional)", expanded=False):
    workspace_mode = st.radio(
        "Tenancy",
        ["Shared (default)", "Nitro Enclaves", "Dedicated Instances", "Dedicated Hosts"],
        help=f"**Shared**: Standard multi-tenant cloud. Your instance runs on shared hardware.\n\n**Nitro Enclaves**: {GLOSSARY['nitro_enclave']}\n\n**Dedicated Instances**: {GLOSSARY['dedicated_instance']}\n\n**Dedicated Hosts**: {GLOSSARY['dedicated_host']}",
    )

    workspace_key = "shared"
    dedicated_host_rate = 0.0

    if workspace_mode == "Nitro Enclaves":
        workspace_key = "nitro_enclave"
        st.info("No additional cost. Nitro Enclaves do **not** support GPU passthrough — CPU/memory isolation only.")
    elif workspace_mode == "Dedicated Instances":
        workspace_key = "dedicated_instance"
        st.caption("~10% premium applied to all fleet hourly rates.")
    elif workspace_mode == "Dedicated Hosts":
        workspace_key = "dedicated_host"
        dedicated_host_rate = st.number_input(
            "Dedicated Host hourly rate ($)", value=25.0, step=1.0,
            help="Fixed rate per host. Check AWS pricing for your instance family.",
        )

# ---------------------------------------------------------------------------
# Sidebar — AWS Direct Connect
# ---------------------------------------------------------------------------
with st.sidebar.expander("AWS Direct Connect (optional)", expanded=False):
    enable_dx = st.checkbox("Add Direct Connect costs",
                            help="Dedicated network connection between your on-prem data center and AWS. Required for low-latency, high-bandwidth hybrid workloads like gradient transport.")
    dx_monthly = 0.0
    if enable_dx:
        dx_port = st.selectbox("Port Speed", list(DIRECT_CONNECT_PRICING["port_hourly"].keys()),
                               index=1,
                               help="Dedicated bandwidth between your facility and AWS. 10 Gbps handles most training workloads. 100 Gbps for large-scale distributed training.")
        dx_outbound = st.slider("Monthly Outbound Data (TB)", 0.0, 100.0, 1.0, 0.5,
                                help="Data leaving AWS back to your on-prem systems. Inbound to AWS is free. Gradient compression can reduce volumes 10-100x.")
        dx_monthly = calc_direct_connect_monthly(dx_port, dx_outbound * 1000)  # TB -> GB
        st.metric("Direct Connect Monthly", f"${dx_monthly:,.0f}")

# ---------------------------------------------------------------------------
# Sidebar — Analysis Settings
# ---------------------------------------------------------------------------
with st.sidebar.expander("Analysis Settings", expanded=False):
    horizon_months = st.slider("Analysis Horizon (months)", 12, 60, 36)
    show_hybrid = st.checkbox("Include hybrid scenario", value=True,
                              help="Hybrid = on-prem hardware costs + cloud fleet costs combined.")

# ---------------------------------------------------------------------------
# Build fleet entries for computation
# ---------------------------------------------------------------------------
fleet_entries = []
for entry in st.session_state["fleet"]:
    if entry["usage_pattern"] == "24/7":
        hours = 24.0
    elif entry["usage_pattern"].startswith("Business"):
        hours = 10.0 * (5.0 / 7.0)
    else:
        hours = entry.get("custom_hours", 12.0)

    inst = EC2_INSTANCES[entry["instance_type"]]
    base_rate = inst[PRICING_TIERS[entry["pricing_tier"]]]

    # Live pricing override
    rate_override = None
    if live_prices:
        if entry["pricing_tier"] == "Spot":
            rate_override = live_prices["spot"].get(entry["instance_type"])
        elif entry["pricing_tier"] == "On-Demand":
            rate_override = live_prices["on_demand"].get(entry["instance_type"])

    effective_rate = rate_override if rate_override is not None else base_rate

    # Apply secure workspace modifier
    effective_rate = apply_secure_workspace(effective_rate, workspace_key, dedicated_host_rate)

    fleet_entries.append({
        "instance_type": entry["instance_type"],
        "count": entry["count"],
        "pricing_tier": entry["pricing_tier"],
        "hours_per_day": hours,
        "hourly_rate_override": effective_rate,
    })

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
capex = calc_onprem_capex(gpu_count, gpu_unit_cost, chassis_cost, ram_cost, storage_cost)
onprem_monthly = calc_onprem_monthly_opex(power_kw, electricity, rack_monthly)
fleet_monthly = calc_fleet_monthly(fleet_entries)
cloud_monthly = fleet_monthly + eks_monthly + dx_monthly
breakeven = calc_breakeven_months(capex, onprem_monthly, cloud_monthly)
hybrid_monthly = onprem_monthly + cloud_monthly if show_hybrid else None

# Store computed values for other pages
st.session_state["computed"] = {
    "capex": capex, "onprem_monthly": onprem_monthly,
    "cloud_monthly": cloud_monthly, "fleet_monthly": fleet_monthly,
    "eks_monthly": eks_monthly, "dx_monthly": dx_monthly,
    "breakeven": breakeven, "hybrid_monthly": hybrid_monthly,
    "fleet_entries": fleet_entries, "gpu_model": gpu_model,
    "gpu_count": gpu_count, "gpu_info": gpu_info,
    "chassis_name": chassis_name, "horizon_months": horizon_months,
    "power_kw": power_kw, "electricity": electricity,
    "rack_monthly": rack_monthly, "show_hybrid": show_hybrid,
    "chassis_cost": chassis_cost, "gpu_unit_cost": gpu_unit_cost,
    "ram_cost": ram_cost, "storage_cost": storage_cost,
}

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Hybrid Infrastructure Cost Modeler")
st.caption("Compare on-prem GPU servers against AWS EC2 fleets — interactive CapEx vs OpEx analysis with EKS, secure workspaces, and live pricing.")

# ---------------------------------------------------------------------------
# Section A: Summary metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("On-Prem CapEx", f"${capex:,.0f}", help=GLOSSARY["capex"])
c2.metric("On-Prem Monthly OpEx", f"${onprem_monthly:,.0f}", help=GLOSSARY["opex"])
cloud_label = "Cloud Fleet Monthly"
if eks_monthly > 0:
    cloud_label = "Cloud Fleet + EKS Monthly"
c3.metric(cloud_label, f"${cloud_monthly:,.0f}", help=GLOSSARY["opex"])
if breakeven is not None:
    c4.metric("Break-Even", f"{breakeven:.1f} months", help=GLOSSARY["breakeven"])
else:
    c4.metric("Break-Even", "Never (cloud cheaper)", help=GLOSSARY["breakeven"])

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
        lines = ["**Cloud Fleet**"]
        for entry in st.session_state["fleet"]:
            inst = EC2_INSTANCES[entry["instance_type"]]
            lines.append(
                f"- {entry['count']}x {entry['instance_type']} "
                f"({inst['gpu_count']}x {inst['gpu_model']}) — {entry['pricing_tier']}, {entry['usage_pattern']}"
            )
        lines.append(f"- Fleet total: ${fleet_monthly:,.0f}/mo | {calc_fleet_vram(fleet_entries)} GB VRAM")
        if eks_monthly > 0:
            lines.append(f"- EKS: ${eks_monthly:,.0f}/mo")
        if workspace_key != "shared":
            lines.append(f"- Tenancy: {workspace_mode}")
        st.markdown("\n".join(lines))

st.divider()

# ---------------------------------------------------------------------------
# Section B: Break-even timeline
# ---------------------------------------------------------------------------
st.subheader("Cumulative Cost Over Time", anchor=False)

timeline = build_timeline_df(capex, onprem_monthly, cloud_monthly, horizon_months, hybrid_monthly)

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(
    x=timeline["month"], y=timeline["onprem_cumulative"],
    name="On-Prem", line=dict(color=COLORS["onprem"], width=3),
    hovertemplate="Month %{x}<br>$%{y:,.0f}<extra>On-Prem</extra>",
))
cloud_trace_name = "Cloud Fleet + EKS" if eks_monthly > 0 else "Cloud Fleet"
fig_timeline.add_trace(go.Scatter(
    x=timeline["month"], y=timeline["cloud_cumulative"],
    name=cloud_trace_name, line=dict(color=COLORS["cloud"], width=3),
    hovertemplate="Month %{x}<br>$%{y:,.0f}<extra>" + cloud_trace_name + "</extra>",
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

fig_timeline.update_layout(**PLOTLY_LAYOUT, xaxis_title="Months",
                           yaxis_title="Cumulative Cost ($)", yaxis_tickformat="$,.0f")
st.plotly_chart(fig_timeline, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------------------
# Section C: Monthly cost breakdown
# ---------------------------------------------------------------------------
st.subheader("Monthly Cost Breakdown", anchor=False)

amort_months = max(horizon_months, 1)
onprem_amortized = capex / amort_months

categories = ["On-Prem", "Cloud Fleet"]
amort_values = [onprem_amortized, 0]
opex_values = [onprem_monthly, cloud_monthly]

if show_hybrid:
    categories.append("Hybrid")
    amort_values.append(onprem_amortized)
    opex_values.append(onprem_monthly + cloud_monthly)

fig_monthly = go.Figure()
fig_monthly.add_trace(go.Bar(
    x=categories, y=amort_values, name="CapEx Amortization",
    marker_color=COLORS["onprem_light"],
    hovertemplate="%{x}<br>$%{y:,.0f}/mo<extra>CapEx Amort.</extra>",
))
fig_monthly.add_trace(go.Bar(
    x=categories, y=opex_values, name="Monthly OpEx",
    marker_color=[COLORS["onprem"], COLORS["cloud"], COLORS["hybrid"]][:len(categories)],
    hovertemplate="%{x}<br>$%{y:,.0f}/mo<extra>OpEx</extra>",
))
fig_monthly.update_layout(**PLOTLY_LAYOUT, barmode="stack",
                          yaxis_title="Monthly Cost ($)", yaxis_tickformat="$,.0f")
st.plotly_chart(fig_monthly, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------------------
# Section D: VRAM comparison
# ---------------------------------------------------------------------------
st.subheader("VRAM Comparison", anchor=False)

# Fleet instances in use
fleet_types = {e["instance_type"] for e in st.session_state["fleet"]}

vram_labels = [f"On-Prem ({gpu_count}x {gpu_model})"]
vram_values = [gpu_count * gpu_info["vram_gb"]]
vram_colors = [COLORS["onprem"]]

# Fleet total
fleet_vram_total = calc_fleet_vram(fleet_entries)
if len(st.session_state["fleet"]) > 1 or st.session_state["fleet"][0]["count"] > 1:
    vram_labels.append(f"Fleet Total ({len(fleet_entries)} entries)")
    vram_values.append(fleet_vram_total)
    vram_colors.append(COLORS["hybrid"])

for name, inst in EC2_INSTANCES.items():
    vram_labels.append(f"{name} ({inst['gpu_count']}x {inst['gpu_model']})")
    vram_values.append(inst["vram_gb"])
    vram_colors.append(COLORS["cloud"] if name in fleet_types else COLORS["cloud_light"])

fig_vram = go.Figure(go.Bar(
    y=vram_labels, x=vram_values, orientation="h",
    marker_color=vram_colors,
    text=[f"{v} GB" for v in vram_values],
    textposition="outside",
    hovertemplate="%{y}<br>%{x} GB<extra></extra>",
))
fig_vram.update_layout(**PLOTLY_LAYOUT, xaxis_title="Total VRAM (GB)",
                       height=max(300, len(vram_labels) * 35))
st.plotly_chart(fig_vram, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------------------
# Section E: TCO table + chart
# ---------------------------------------------------------------------------
st.subheader("Total Cost of Ownership", anchor=False)

tco = build_tco_table(capex, onprem_monthly, cloud_monthly, hybrid_monthly)
tco_display = tco.copy()
for col in tco_display.columns:
    if "TCO" in col:
        tco_display[col] = tco_display[col].apply(lambda v: f"${v:,.0f}")
st.dataframe(tco_display, use_container_width=True, hide_index=True)

fig_tco = go.Figure()
fig_tco.add_trace(go.Bar(x=tco["Horizon"], y=tco["On-Prem TCO"], name="On-Prem",
                         marker_color=COLORS["onprem"],
                         hovertemplate="%{x}<br>$%{y:,.0f}<extra>On-Prem</extra>"))
fig_tco.add_trace(go.Bar(x=tco["Horizon"], y=tco["Cloud TCO"], name="Cloud Fleet",
                         marker_color=COLORS["cloud"],
                         hovertemplate="%{x}<br>$%{y:,.0f}<extra>Cloud Fleet</extra>"))
if show_hybrid and "Hybrid TCO" in tco.columns:
    fig_tco.add_trace(go.Bar(x=tco["Horizon"], y=tco["Hybrid TCO"], name="Hybrid",
                             marker_color=COLORS["hybrid"],
                             hovertemplate="%{x}<br>$%{y:,.0f}<extra>Hybrid</extra>"))
fig_tco.update_layout(**PLOTLY_LAYOUT, barmode="group",
                      yaxis_title="Total Cost ($)", yaxis_tickformat="$,.0f")
st.plotly_chart(fig_tco, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------------------
# Section F: Export
# ---------------------------------------------------------------------------
st.subheader("Export", anchor=False)

csv = timeline.to_csv(index=False)
st.download_button(label="Download timeline data (CSV)", data=csv,
                   file_name="cost_timeline.csv", mime="text/csv")
st.caption("Use the camera icon in the top-right of each chart to export as PNG.")
