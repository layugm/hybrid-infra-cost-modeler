# Hybrid Infrastructure Cost Modeler

Interactive Streamlit dashboard for comparing on-prem GPU server costs (CapEx) against AWS EC2 cloud compute (OpEx). Helps answer: **should you buy or rent your GPU infrastructure?**

![Python](https://img.shields.io/badge/python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red) ![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **12 GPU models** — V100, A100 (40/80GB), L40, L40S, H100 (PCIe/SXM), RTX A5000, RTX A6000, MI210, MI250
- **6 server chassis** — Dell R750xa/R760xa, HPE DL380a Gen10+/Gen11, Supermicro 4U, Custom
- **12 EC2 instance types** — G5, G6, G6e, P4d, P4de, P5 families
- **Live AWS pricing** — real-time spot and on-demand rates via boto3 (optional)
- **Break-even analysis** — when does buying hardware pay for itself vs renting cloud?
- **TCO comparison** — 1, 2, and 3-year total cost of ownership
- **VRAM comparison** — side-by-side across all configurations
- **Auto power estimation** — calculates system draw from GPU TDP
- **Export** — CSV data download + PNG chart export via Plotly

## Quick Start

```bash
git clone https://github.com/layugm/hybrid-infra-cost-modeler.git
cd hybrid-infra-cost-modeler
pip install -r requirements.txt
streamlit run modeler.py
```

Opens at `http://localhost:8501`.

## Live AWS Pricing (Optional)

Toggle "Live AWS pricing" in the sidebar to fetch real-time spot and on-demand rates. Requires AWS credentials:

```bash
aws configure
# or
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Falls back to static pricing (March 2026 snapshot) if credentials aren't configured or boto3 isn't installed.

## Project Structure

```
modeler.py          # Main dashboard — UI, charts, sidebar controls
data.py             # GPU/chassis/EC2 catalogs, cost functions, AWS pricing API
requirements.txt    # Python dependencies
```

## How It Works

**On-prem costs** are split into:
- **CapEx** — one-time hardware purchase (chassis + GPUs + RAM + storage)
- **Monthly OpEx** — electricity (auto-estimated from GPU TDP) + rack rental

**Cloud costs** are calculated from:
- EC2 hourly rate (on-demand, spot, or 1-year reserved)
- Usage pattern (24/7, business hours, or custom hours/day)

**Break-even** = CapEx / (cloud monthly - on-prem monthly OpEx)

The hybrid scenario models owning on-prem hardware for privacy-sensitive workloads while renting cloud compute for burst capacity.

## Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"Create app"**
3. Fill in:
   - **Repository**: `layugm/hybrid-infra-cost-modeler`
   - **Branch**: `main`
   - **Main file path**: `modeler.py`
4. Pick a subdomain and click **Deploy**

The app auto-redeploys on every push to `main`.

To enable live AWS pricing on the deployed app, add your credentials via **Settings > Secrets** in the Streamlit Cloud dashboard:

```toml
[aws]
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_DEFAULT_REGION = "us-east-1"
```

## Adding GPUs or Instances

Edit the catalogs in `data.py`:

```python
# Add a GPU
GPU_CATALOG["RTX 4090 24GB"] = {
    "price_default": 1800, "price_low": 1500, "price_high": 2200,
    "vram_gb": 24, "tdp_w": 450,
}

# Add an EC2 instance
EC2_INSTANCES["p5e.48xlarge"] = {
    "gpu_model": "H200", "gpu_count": 8, "vram_gb": 1128,
    "vcpus": 192, "ram_gb": 2048,
    "on_demand_hr": 65.00, "spot_hr": 35.00, "reserved_1yr_hr": 28.00,
}
```

The dashboard picks up new entries automatically.

## License

MIT
