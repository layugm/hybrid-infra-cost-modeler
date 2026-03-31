"""AI Advisor — Claude-powered infrastructure consultant."""

import streamlit as st

from data import EC2_INSTANCES, GPU_CATALOG, GLOSSARY, calc_fleet_vram

st.set_page_config(page_title="AI Advisor", page_icon=":speech_balloon:", layout="wide")

st.title("AI Advisor")
st.caption("Ask questions about your infrastructure configuration. Powered by Claude.")

# ---------------------------------------------------------------------------
# API key input
# ---------------------------------------------------------------------------
if "claude_api_key" not in st.session_state:
    st.session_state["claude_api_key"] = ""

api_key = st.text_input(
    "Claude API Key",
    type="password",
    value=st.session_state["claude_api_key"],
    help="Your key is stored only in this browser session. It is never saved to disk, logged, or sent anywhere except the Anthropic API.",
    placeholder="sk-ant-...",
)
st.session_state["claude_api_key"] = api_key

if not api_key:
    st.info(
        "Enter your Claude API key above to enable the AI advisor. "
        "You can get one at [console.anthropic.com](https://console.anthropic.com). "
        "Your key stays in your browser session only — it is never stored or logged."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Check for computed data
# ---------------------------------------------------------------------------
if "computed" not in st.session_state:
    st.warning("Configure your infrastructure on the main **Cost Modeler** page first, then return here.")
    st.stop()

c = st.session_state["computed"]

# ---------------------------------------------------------------------------
# Build system prompt with current config context
# ---------------------------------------------------------------------------

def build_system_prompt():
    fleet_lines = []
    for entry in c["fleet_entries"]:
        inst = EC2_INSTANCES[entry["instance_type"]]
        fleet_lines.append(
            f"  - {entry['count']}x {entry['instance_type']} "
            f"({inst['gpu_count']}x {inst['gpu_model']}, {inst['vram_gb']}GB VRAM) "
            f"@ ${entry.get('hourly_rate_override', 0):.2f}/hr, "
            f"{entry['hours_per_day']:.1f} hrs/day"
        )
    fleet_text = "\n".join(fleet_lines)
    fleet_vram = calc_fleet_vram(c["fleet_entries"])

    return f"""You are an AI infrastructure cost advisor. You help users make informed decisions about on-prem vs cloud GPU infrastructure.

CURRENT CONFIGURATION:
- On-Prem: {c['chassis_name']} with {c['gpu_count']}x {c['gpu_model']} ({c['gpu_count'] * c['gpu_info']['vram_gb']}GB total VRAM)
- On-Prem CapEx: ${c['capex']:,.0f}
- On-Prem Monthly OpEx: ${c['onprem_monthly']:,.0f}/mo (power: {c['power_kw']:.1f}kW @ ${c['electricity']:.2f}/kWh)
- Cloud Fleet:
{fleet_text}
- Cloud Fleet Monthly: ${c['fleet_monthly']:,.0f}/mo
- Cloud Fleet VRAM: {fleet_vram}GB
- EKS Monthly: ${c['eks_monthly']:,.0f}/mo
- Direct Connect Monthly: ${c['dx_monthly']:,.0f}/mo
- Total Cloud Monthly: ${c['cloud_monthly']:,.0f}/mo
- Break-Even: {f"{c['breakeven']:.1f} months" if c['breakeven'] else "Never (cloud cheaper)"}
- Analysis Horizon: {c['horizon_months']} months

GLOSSARY (use these definitions when explaining concepts):
{chr(10).join(f'- {k}: {v}' for k, v in GLOSSARY.items())}

GUIDELINES:
- Give concise, actionable answers
- Reference the specific numbers from the current configuration
- When recommending changes, explain the cost impact
- Use plain language — the user may not be deeply technical
- If asked about something outside infrastructure costs, politely redirect"""

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Quick-start prompts
if not st.session_state["chat_history"]:
    st.markdown("**Quick start — click a question:**")
    prompts_col1, prompts_col2 = st.columns(2)
    quick_prompts = [
        "Is my current configuration cost-effective?",
        "How can I reduce my cloud spending?",
        "Explain the break-even analysis to me",
        "What happens if GPU prices drop 20%?",
    ]
    for i, prompt in enumerate(quick_prompts):
        col = prompts_col1 if i % 2 == 0 else prompts_col2
        if col.button(prompt, key=f"quick_{i}"):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            st.rerun()

# Display chat history
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about your infrastructure configuration..."):
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

# Generate response for the last user message
if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
    try:
        import anthropic
    except ImportError:
        st.error("The `anthropic` package is not installed. Run: `pip install anthropic`")
        st.stop()

    try:
        client = anthropic.Anthropic(api_key=api_key)

        with st.chat_message("assistant"):
            # Build messages for API (exclude system prompt from messages)
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["chat_history"]
            ]

            with st.spinner("Thinking..."):
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=build_system_prompt(),
                    messages=api_messages,
                )

            assistant_msg = response.content[0].text
            st.markdown(assistant_msg)
            st.session_state["chat_history"].append({"role": "assistant", "content": assistant_msg})

    except anthropic.AuthenticationError:
        st.error("Invalid API key. Check your key at [console.anthropic.com](https://console.anthropic.com).")
    except anthropic.RateLimitError:
        st.error("Rate limited. Wait a moment and try again.")
    except Exception as e:
        st.error(f"Error: {e}")

# Clear chat button
if st.session_state["chat_history"]:
    if st.button("Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()
