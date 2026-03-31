"""Cost model for hybrid on-prem vs cloud GPU infrastructure comparison."""

import logging
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# On-prem GPU catalog (used/refurbished market pricing, March 2026)
# ---------------------------------------------------------------------------
GPU_CATALOG = {
    # NVIDIA — Data Center
    "V100 16GB": {"price_default": 1200, "price_low": 800, "price_high": 2000, "vram_gb": 16, "tdp_w": 300},
    "V100 32GB": {"price_default": 1700, "price_low": 1200, "price_high": 2500, "vram_gb": 32, "tdp_w": 300},
    "A100 40GB": {"price_default": 4500, "price_low": 4000, "price_high": 8000, "vram_gb": 40, "tdp_w": 300},
    "A100 80GB": {"price_default": 5500, "price_low": 4000, "price_high": 9000, "vram_gb": 80, "tdp_w": 300},
    "L40": {"price_default": 5000, "price_low": 4000, "price_high": 6000, "vram_gb": 48, "tdp_w": 300},
    "L40S": {"price_default": 8000, "price_low": 7500, "price_high": 9000, "vram_gb": 48, "tdp_w": 350},
    "H100 80GB PCIe": {"price_default": 22000, "price_low": 18000, "price_high": 32000, "vram_gb": 80, "tdp_w": 350},
    "H100 80GB SXM": {"price_default": 28000, "price_low": 22000, "price_high": 40000, "vram_gb": 80, "tdp_w": 700},
    # NVIDIA — Professional / Workstation
    "RTX A5000 24GB": {"price_default": 3300, "price_low": 2500, "price_high": 4000, "vram_gb": 24, "tdp_w": 230},
    "RTX A6000 48GB": {"price_default": 4500, "price_low": 3500, "price_high": 6000, "vram_gb": 48, "tdp_w": 300},
    # AMD — Data Center
    "MI210 64GB": {"price_default": 1000, "price_low": 600, "price_high": 1500, "vram_gb": 64, "tdp_w": 300},
    "MI250 128GB": {"price_default": 5500, "price_low": 4000, "price_high": 7000, "vram_gb": 128, "tdp_w": 500},
}

# ---------------------------------------------------------------------------
# Server chassis catalog (refurbished pricing, March 2026)
# ---------------------------------------------------------------------------
CHASSIS_CATALOG = {
    "Dell R750xa (2U, 4 GPU)": {"price_default": 7000, "price_low": 3000, "price_high": 12000, "max_gpus": 4},
    "Dell R760xa (2U, 4 GPU)": {"price_default": 12000, "price_low": 8000, "price_high": 18000, "max_gpus": 4},
    "HPE DL380a Gen10+ (2U, 4 GPU)": {"price_default": 6000, "price_low": 4000, "price_high": 10000, "max_gpus": 4},
    "HPE DL380a Gen11 (2U, 4 GPU)": {"price_default": 11000, "price_low": 8000, "price_high": 15000, "max_gpus": 4},
    "Supermicro 4U GPU Server (8 GPU)": {"price_default": 5000, "price_low": 3000, "price_high": 8000, "max_gpus": 8},
    "Custom / Other": {"price_default": 5000, "price_low": 0, "price_high": 30000, "max_gpus": 8},
}

ONPREM_DEFAULTS = {
    "ram_cost": 2500,
    "storage_cost": 2500,
    "gpu_count": 4,
    "system_base_power_kw": 0.5,  # chassis + CPU + fans (without GPUs)
    "electricity_kwh": 0.12,
    "rack_monthly": 0,
}

# ---------------------------------------------------------------------------
# EC2 instance catalog (us-east-1 pricing, March 2026)
# ---------------------------------------------------------------------------
EC2_INSTANCES = {
    # G5 — NVIDIA A10G (Ampere)
    "g5.xlarge": {"gpu_model": "A10G", "gpu_count": 1, "vram_gb": 24, "vcpus": 4, "ram_gb": 16,
                  "on_demand_hr": 1.006, "spot_hr": 0.434, "reserved_1yr_hr": 0.634},
    "g5.12xlarge": {"gpu_model": "A10G", "gpu_count": 4, "vram_gb": 96, "vcpus": 48, "ram_gb": 192,
                    "on_demand_hr": 5.672, "spot_hr": 2.489, "reserved_1yr_hr": 3.573},
    "g5.48xlarge": {"gpu_model": "A10G", "gpu_count": 8, "vram_gb": 192, "vcpus": 192, "ram_gb": 768,
                    "on_demand_hr": 16.288, "spot_hr": 7.402, "reserved_1yr_hr": 10.261},
    # G6 — NVIDIA L4 (Ada Lovelace)
    "g6.xlarge": {"gpu_model": "L4", "gpu_count": 1, "vram_gb": 24, "vcpus": 4, "ram_gb": 16,
                  "on_demand_hr": 0.805, "spot_hr": 0.345, "reserved_1yr_hr": 0.524},
    "g6.12xlarge": {"gpu_model": "L4", "gpu_count": 4, "vram_gb": 96, "vcpus": 48, "ram_gb": 192,
                    "on_demand_hr": 4.602, "spot_hr": 2.153, "reserved_1yr_hr": 2.996},
    "g6.48xlarge": {"gpu_model": "L4", "gpu_count": 8, "vram_gb": 192, "vcpus": 192, "ram_gb": 768,
                    "on_demand_hr": 13.35, "spot_hr": 6.388, "reserved_1yr_hr": 8.691},
    # G6e — NVIDIA L40S (Ada Lovelace)
    "g6e.xlarge": {"gpu_model": "L40S", "gpu_count": 1, "vram_gb": 48, "vcpus": 4, "ram_gb": 32,
                   "on_demand_hr": 1.862, "spot_hr": 0.795, "reserved_1yr_hr": 1.174},
    "g6e.12xlarge": {"gpu_model": "L40S", "gpu_count": 4, "vram_gb": 192, "vcpus": 48, "ram_gb": 384,
                     "on_demand_hr": 10.493, "spot_hr": 4.471, "reserved_1yr_hr": 6.611},
    "g6e.48xlarge": {"gpu_model": "L40S", "gpu_count": 8, "vram_gb": 384, "vcpus": 192, "ram_gb": 1536,
                     "on_demand_hr": 30.13, "spot_hr": 9.82, "reserved_1yr_hr": 18.98},
    # P4d — NVIDIA A100 40GB (Ampere)
    "p4d.24xlarge": {"gpu_model": "A100 40GB", "gpu_count": 8, "vram_gb": 320, "vcpus": 96, "ram_gb": 1152,
                     "on_demand_hr": 21.96, "spot_hr": 10.10, "reserved_1yr_hr": 13.92},
    # P4de — NVIDIA A100 80GB (Ampere)
    "p4de.24xlarge": {"gpu_model": "A100 80GB", "gpu_count": 8, "vram_gb": 640, "vcpus": 96, "ram_gb": 1152,
                      "on_demand_hr": 27.45, "spot_hr": 17.46, "reserved_1yr_hr": 17.40},
    # P5 — NVIDIA H100 (Hopper)
    "p5.48xlarge": {"gpu_model": "H100", "gpu_count": 8, "vram_gb": 640, "vcpus": 192, "ram_gb": 2048,
                    "on_demand_hr": 55.04, "spot_hr": 30.455, "reserved_1yr_hr": 23.777},
}

PRICING_TIERS = {
    "On-Demand": "on_demand_hr",
    "Spot": "spot_hr",
    "1-Year Reserved": "reserved_1yr_hr",
}

AWS_REGIONS = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "eu-west-1": "Europe (Ireland)",
    "eu-central-1": "Europe (Frankfurt)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
}

DAYS_PER_MONTH = 30.44

# ---------------------------------------------------------------------------
# Cost computation functions
# ---------------------------------------------------------------------------

def calc_onprem_capex(
    gpu_count: int,
    gpu_unit_cost: float,
    chassis: float,
    ram: float = ONPREM_DEFAULTS["ram_cost"],
    storage: float = ONPREM_DEFAULTS["storage_cost"],
) -> float:
    return chassis + (gpu_count * gpu_unit_cost) + ram + storage


def calc_onprem_monthly_opex(
    power_kw: float,
    cost_per_kwh: float = ONPREM_DEFAULTS["electricity_kwh"],
    rack_monthly: float = ONPREM_DEFAULTS["rack_monthly"],
) -> float:
    hours_per_month = 24 * DAYS_PER_MONTH
    return (power_kw * hours_per_month * cost_per_kwh) + rack_monthly


def estimate_system_power(gpu_count: int, gpu_tdp_w: int,
                          base_kw: float = ONPREM_DEFAULTS["system_base_power_kw"]) -> float:
    """Estimate total system power draw in kW."""
    return base_kw + (gpu_count * gpu_tdp_w / 1000)


def calc_cloud_monthly(hourly_rate: float, hours_per_day: float) -> float:
    return hourly_rate * hours_per_day * DAYS_PER_MONTH


def calc_breakeven_months(
    capex: float, onprem_monthly: float, cloud_monthly: float
) -> float | None:
    delta = cloud_monthly - onprem_monthly
    if delta <= 0:
        return None
    return capex / delta


def build_timeline_df(
    capex: float,
    onprem_monthly: float,
    cloud_monthly: float,
    months: int = 36,
    hybrid_monthly: float | None = None,
) -> pd.DataFrame:
    data = []
    for m in range(months + 1):
        row = {
            "month": m,
            "onprem_cumulative": capex + (onprem_monthly * m),
            "cloud_cumulative": cloud_monthly * m,
        }
        if hybrid_monthly is not None:
            row["hybrid_cumulative"] = capex + (hybrid_monthly * m)
        data.append(row)
    return pd.DataFrame(data)


def build_tco_table(
    capex: float,
    onprem_monthly: float,
    cloud_monthly: float,
    hybrid_monthly: float | None = None,
) -> pd.DataFrame:
    rows = []
    for years in [1, 2, 3]:
        m = years * 12
        onprem_tco = capex + (onprem_monthly * m)
        cloud_tco = cloud_monthly * m
        row = {
            "Horizon": f"{years} Year{'s' if years > 1 else ''}",
            "On-Prem TCO": onprem_tco,
            "Cloud TCO": cloud_tco,
        }
        if hybrid_monthly is not None:
            hybrid_tco = capex + (hybrid_monthly * m)
            row["Hybrid TCO"] = hybrid_tco
            costs = {"On-Prem": onprem_tco, "Cloud": cloud_tco, "Hybrid": hybrid_tco}
        else:
            costs = {"On-Prem": onprem_tco, "Cloud": cloud_tco}
        row["Cheapest"] = min(costs, key=costs.get)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EKS pricing (March 2026)
# ---------------------------------------------------------------------------
EKS_PRICING = {
    "control_plane_hr": 0.10,
    "control_plane_extended_hr": 0.60,
    "hybrid_tiers": [
        (576_000, 0.020),
        (11_520_000, 0.014),
        (float("inf"), 0.006),
    ],
    "anywhere_monthly_1yr": 2000,
    "anywhere_monthly_3yr": 1500,
}

HOURS_PER_MONTH = 24 * DAYS_PER_MONTH  # ~730


def calc_eks_control_plane_monthly(
    cluster_count: int = 1,
    extended_support: bool = False,
) -> float:
    rate = EKS_PRICING["control_plane_extended_hr" if extended_support else "control_plane_hr"]
    return rate * HOURS_PER_MONTH * cluster_count


def calc_eks_hybrid_monthly(vcpu_count: int, hours_per_month: float = HOURS_PER_MONTH) -> float:
    total_vcpu_hours = vcpu_count * hours_per_month
    cost = 0.0
    remaining = total_vcpu_hours
    prev_cap = 0
    for cap, rate in EKS_PRICING["hybrid_tiers"]:
        tier_hours = min(remaining, cap - prev_cap)
        cost += tier_hours * rate
        remaining -= tier_hours
        prev_cap = cap
        if remaining <= 0:
            break
    return cost


def calc_eks_anywhere_monthly(term_years: int = 1, cluster_count: int = 1) -> float:
    key = "anywhere_monthly_3yr" if term_years == 3 else "anywhere_monthly_1yr"
    return EKS_PRICING[key] * cluster_count


def calc_eks_total_monthly(
    cluster_count: int = 1,
    extended_support: bool = False,
    hybrid_vcpus: int = 0,
    anywhere: bool = False,
    anywhere_term: int = 1,
) -> float:
    total = calc_eks_control_plane_monthly(cluster_count, extended_support)
    if hybrid_vcpus > 0:
        total += calc_eks_hybrid_monthly(hybrid_vcpus)
    if anywhere:
        total += calc_eks_anywhere_monthly(anywhere_term, cluster_count)
    return total


# ---------------------------------------------------------------------------
# Fleet aggregation
# ---------------------------------------------------------------------------

def calc_fleet_monthly(fleet: list[dict]) -> float:
    """Sum cloud cost across all fleet entries.

    Each entry: {"instance_type": str, "count": int, "pricing_tier": str,
                 "hours_per_day": float, "hourly_rate_override": float | None}
    """
    total = 0.0
    for entry in fleet:
        inst = EC2_INSTANCES[entry["instance_type"]]
        if entry.get("hourly_rate_override") is not None:
            rate = entry["hourly_rate_override"]
        else:
            rate = inst[PRICING_TIERS[entry["pricing_tier"]]]
        total += calc_cloud_monthly(rate, entry["hours_per_day"]) * entry["count"]
    return total


def calc_fleet_vram(fleet: list[dict]) -> int:
    """Sum total VRAM across all fleet entries."""
    total = 0
    for entry in fleet:
        inst = EC2_INSTANCES[entry["instance_type"]]
        total += inst["vram_gb"] * entry["count"]
    return total


# ---------------------------------------------------------------------------
# Secure workspace modifiers
# ---------------------------------------------------------------------------

def apply_secure_workspace(
    base_hourly: float,
    mode: str,
    dedicated_host_hourly: float = 0.0,
) -> float:
    """Apply tenancy pricing modifier.

    Modes: "shared", "nitro_enclave", "dedicated_instance", "dedicated_host"
    """
    if mode == "dedicated_instance":
        return base_hourly * 1.10
    elif mode == "dedicated_host":
        return dedicated_host_hourly
    return base_hourly  # shared and nitro_enclave have no cost change


# ---------------------------------------------------------------------------
# Live AWS pricing via boto3
# ---------------------------------------------------------------------------

def _instance_type_map() -> dict[str, str]:
    """Build display_name -> EC2 API instance type mapping dynamically."""
    return {k: k for k in EC2_INSTANCES}


def fetch_spot_prices(region: str = "us-east-1") -> dict[str, float | None]:
    """Fetch current EC2 spot prices for all cataloged instance types."""
    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed — skipping live spot prices")
        return {k: None for k in EC2_INSTANCES}

    try:
        ec2 = boto3.client("ec2", region_name=region)
        result = {}
        for instance_type in EC2_INSTANCES:
            resp = ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=1,
            )
            history = resp.get("SpotPriceHistory", [])
            if history:
                result[instance_type] = float(history[0]["SpotPrice"])
            else:
                result[instance_type] = None
        return result
    except Exception as e:
        log.warning("Failed to fetch spot prices: %s", e)
        return {k: None for k in EC2_INSTANCES}


def fetch_on_demand_prices(region: str = "us-east-1") -> dict[str, float | None]:
    """Fetch current EC2 on-demand prices from the AWS Price List API."""
    try:
        import boto3
        import json
    except ImportError:
        log.warning("boto3 not installed — skipping live on-demand prices")
        return {k: None for k in EC2_INSTANCES}

    try:
        pricing = boto3.client("pricing", region_name="us-east-1")
        location = AWS_REGIONS.get(region, "US East (N. Virginia)")
        result = {}

        for instance_type in EC2_INSTANCES:
            resp = pricing.get_products(
                ServiceCode="AmazonEC2",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                    {"Type": "TERM_MATCH", "Field": "location", "Value": location},
                    {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                    {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                    {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                    {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                ],
                MaxResults=1,
            )
            price_list = resp.get("PriceList", [])
            if price_list:
                offer = json.loads(price_list[0])
                terms = offer.get("terms", {}).get("OnDemand", {})
                for term in terms.values():
                    for dim in term.get("priceDimensions", {}).values():
                        price = float(dim["pricePerUnit"]["USD"])
                        if price > 0:
                            result[instance_type] = price
                            break
            if instance_type not in result:
                result[instance_type] = None

        return result
    except Exception as e:
        log.warning("Failed to fetch on-demand prices: %s", e)
        return {k: None for k in EC2_INSTANCES}


def fetch_all_live_prices(region: str = "us-east-1") -> dict:
    """Fetch both spot and on-demand prices. Returns structured dict."""
    spot = fetch_spot_prices(region)
    on_demand = fetch_on_demand_prices(region)
    return {
        "spot": spot,
        "on_demand": on_demand,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "region": region,
    }
