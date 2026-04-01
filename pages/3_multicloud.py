"""Multi-Cloud GPU Price Comparison — AWS vs GCP vs Azure."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data import EC2_INSTANCES, GCP_INSTANCES, AZURE_INSTANCES, DAYS_PER_MONTH

st.set_page_config(page_title="Multi-Cloud", page_icon=":globe_with_meridians:", layout="wide")

st.title("Multi-Cloud GPU Price Comparison", anchor=False)
st.caption("Side-by-side pricing for equivalent GPU instances across AWS, Google Cloud, and Microsoft Azure.")

CLOUD_COLORS = {"AWS": "#FF9900", "GCP": "#4285F4", "Azure": "#0078D4"}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hoverlabel=dict(namelength=-1),
)

# ---------------------------------------------------------------------------
# Build unified catalog
# ---------------------------------------------------------------------------
all_instances = []

for name, inst in EC2_INSTANCES.items():
    all_instances.append({"cloud": "AWS", "instance": name, **inst})

for name, inst in GCP_INSTANCES.items():
    all_instances.append({"instance": name, **inst})

for name, inst in AZURE_INSTANCES.items():
    all_instances.append({"instance": name, **inst})

df = pd.DataFrame(all_instances)

# Normalize GPU model names for matching
gpu_model_map = {
    "A10G": "A10G",
    "L4": "L4",
    "L40S": "L40S",
    "A100 40GB": "A100",
    "A100 80GB": "A100",
    "H100": "H100",
}
df["gpu_family"] = df["gpu_model"].map(gpu_model_map).fillna(df["gpu_model"])

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

available_gpus = sorted(df["gpu_family"].unique())
selected_gpu = col1.selectbox("Filter by GPU family", ["All"] + available_gpus,
                              help="Show only instances with this GPU type across all clouds.")

usage_pattern = col2.selectbox("Monthly projection based on",
                               ["24/7", "Business hours (10h weekdays)", "8 hours/day"],
                               help="Used for the monthly cost column.")

if usage_pattern == "24/7":
    hours_per_day = 24.0
elif usage_pattern.startswith("Business"):
    hours_per_day = 10.0 * (5.0 / 7.0)
else:
    hours_per_day = 8.0

# Filter
filtered = df if selected_gpu == "All" else df[df["gpu_family"] == selected_gpu]

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
st.subheader("Price Comparison", anchor=False)

table_data = []
for _, row in filtered.iterrows():
    monthly_od = row["on_demand_hr"] * hours_per_day * DAYS_PER_MONTH
    monthly_spot = row["spot_hr"] * hours_per_day * DAYS_PER_MONTH
    table_data.append({
        "Cloud": row["cloud"],
        "Instance": row["instance"],
        "GPU": f"{row['gpu_count']}x {row['gpu_model']}",
        "VRAM": f"{row['vram_gb']} GB",
        "On-Demand/hr": f"${row['on_demand_hr']:.2f}",
        "Spot/hr": f"${row['spot_hr']:.2f}",
        "On-Demand/mo": f"${monthly_od:,.0f}",
        "Spot/mo": f"${monthly_spot:,.0f}",
        "Use Case": row.get("use_case", ""),
    })

if table_data:
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
else:
    st.info("No instances match the selected filter.")
    st.stop()

# ---------------------------------------------------------------------------
# On-demand price comparison chart
# ---------------------------------------------------------------------------
st.subheader("On-Demand Hourly Rate by Cloud", anchor=False)

fig_od = go.Figure()
for cloud in ["AWS", "GCP", "Azure"]:
    cloud_df = filtered[filtered["cloud"] == cloud].sort_values("on_demand_hr")
    if cloud_df.empty:
        continue
    labels = [f"{r['instance']}\n{r['gpu_count']}x {r['gpu_model']}" for _, r in cloud_df.iterrows()]
    fig_od.add_trace(go.Bar(
        x=labels,
        y=cloud_df["on_demand_hr"],
        name=cloud,
        marker_color=CLOUD_COLORS[cloud],
        hovertemplate="%{x}<br>$%{y:.2f}/hr<extra>" + cloud + "</extra>",
    ))
fig_od.update_layout(**PLOTLY_LAYOUT, barmode="group",
                     yaxis_title="Hourly Rate ($)", yaxis_tickformat="$.2f",
                     height=450)
st.plotly_chart(fig_od, use_container_width=True)

# ---------------------------------------------------------------------------
# Spot price comparison chart
# ---------------------------------------------------------------------------
st.subheader("Spot / Preemptible Hourly Rate by Cloud", anchor=False)

fig_spot = go.Figure()
for cloud in ["AWS", "GCP", "Azure"]:
    cloud_df = filtered[filtered["cloud"] == cloud].sort_values("spot_hr")
    if cloud_df.empty:
        continue
    labels = [f"{r['instance']}\n{r['gpu_count']}x {r['gpu_model']}" for _, r in cloud_df.iterrows()]
    fig_spot.add_trace(go.Bar(
        x=labels,
        y=cloud_df["spot_hr"],
        name=cloud,
        marker_color=CLOUD_COLORS[cloud],
        hovertemplate="%{x}<br>$%{y:.2f}/hr<extra>" + cloud + "</extra>",
    ))
fig_spot.update_layout(**PLOTLY_LAYOUT, barmode="group",
                       yaxis_title="Hourly Rate ($)", yaxis_tickformat="$.2f",
                       height=450)
st.plotly_chart(fig_spot, use_container_width=True)

# ---------------------------------------------------------------------------
# Monthly cost comparison (best option per GPU per cloud)
# ---------------------------------------------------------------------------
st.subheader("Cheapest Option per GPU Family", anchor=False)
st.caption("Lowest on-demand hourly rate for each GPU type, per cloud provider.")

summary_rows = []
for gpu_fam in sorted(filtered["gpu_family"].unique()):
    gpu_df = filtered[filtered["gpu_family"] == gpu_fam]
    for cloud in ["AWS", "GCP", "Azure"]:
        cloud_gpu = gpu_df[gpu_df["cloud"] == cloud]
        if cloud_gpu.empty:
            continue
        cheapest = cloud_gpu.loc[cloud_gpu["on_demand_hr"].idxmin()]
        monthly = cheapest["on_demand_hr"] * hours_per_day * DAYS_PER_MONTH
        summary_rows.append({
            "GPU Family": gpu_fam,
            "Cloud": cloud,
            "Cheapest Instance": cheapest["instance"],
            "GPUs": f"{int(cheapest['gpu_count'])}x {cheapest['gpu_model']}",
            "On-Demand/hr": f"${cheapest['on_demand_hr']:.2f}",
            "Monthly": f"${monthly:,.0f}",
        })

if summary_rows:
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.caption("Prices are static snapshots (March 2026). Actual rates vary by region and change over time.")
