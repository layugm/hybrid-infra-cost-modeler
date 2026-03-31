"""AI Advisor — LLM-powered infrastructure consultant."""

import streamlit as st

from data import EC2_INSTANCES, GPU_CATALOG, GLOSSARY, calc_fleet_vram

st.set_page_config(page_title="AI Advisor", page_icon=":speech_balloon:", layout="wide")

st.title("AI Advisor", anchor=False)
st.caption("Ask questions about your infrastructure configuration. Supports Claude, OpenAI, and any OpenAI-compatible API.")

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------
PROVIDERS = {
    "Claude (Anthropic)": {
        "placeholder": "sk-ant-...",
        "help": "Get a key at [console.anthropic.com](https://console.anthropic.com)",
        "default_model": "claude-sonnet-4-6",
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"],
    },
    "OpenAI": {
        "placeholder": "sk-...",
        "help": "Get a key at [platform.openai.com](https://platform.openai.com/api-keys)",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
    },
    "OpenAI-Compatible (custom endpoint)": {
        "placeholder": "your-api-key",
        "help": "Works with Groq, Together, Fireworks, Ollama, vLLM, or any OpenAI-compatible API",
        "default_model": "",
        "models": [],
    },
}

provider = st.selectbox("Provider", list(PROVIDERS.keys()),
                        help="All providers use the same system prompt and conversation format. Your API key stays in this browser session only.")
prov = PROVIDERS[provider]

# API key
if "advisor_api_key" not in st.session_state:
    st.session_state["advisor_api_key"] = ""

api_key = st.text_input(
    "API Key", type="password",
    value=st.session_state["advisor_api_key"],
    help="Stored only in this browser session. Never saved to disk or logged.",
    placeholder=prov["placeholder"],
)
st.session_state["advisor_api_key"] = api_key

# Model selection
if provider == "OpenAI-Compatible (custom endpoint)":
    base_url = st.text_input("API Base URL", placeholder="http://localhost:11434/v1",
                             help="The OpenAI-compatible endpoint. For Ollama: http://localhost:11434/v1")
    model = st.text_input("Model name", placeholder="llama3.1:70b",
                          help="The model ID your endpoint serves.")
    # Ollama doesn't need a key
    needs_key = base_url and not base_url.startswith("http://localhost")
else:
    base_url = None
    if prov["models"]:
        model = st.selectbox("Model", prov["models"],
                             index=prov["models"].index(prov["default_model"]))
    else:
        model = prov["default_model"]
    needs_key = True

if needs_key and not api_key:
    st.info(f"Enter your API key above to enable the advisor. {prov['help']}")
    st.stop()

if provider == "OpenAI-Compatible (custom endpoint)" and not base_url:
    st.info("Enter your API base URL above.")
    st.stop()

if not model:
    st.info("Enter a model name above.")
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
# API client factory
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], system: str) -> str:
    """Call the selected LLM provider and return the assistant message."""
    if provider == "Claude (Anthropic)":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    else:
        # OpenAI and OpenAI-compatible use the same SDK
        import openai
        client_kwargs = {"api_key": api_key or "ollama"}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return response.choices[0].message.content


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
        with st.chat_message("assistant"):
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["chat_history"]
            ]
            with st.spinner("Thinking..."):
                assistant_msg = call_llm(api_messages, build_system_prompt())
            st.markdown(assistant_msg)
            st.session_state["chat_history"].append({"role": "assistant", "content": assistant_msg})
    except Exception as e:
        error_str = str(e).lower()
        if "auth" in error_str or "api key" in error_str or "401" in error_str:
            st.error(f"Authentication failed. Check your API key. {prov['help']}")
        elif "rate" in error_str or "429" in error_str:
            st.error("Rate limited. Wait a moment and try again.")
        elif "connection" in error_str or "connect" in error_str:
            st.error(f"Could not connect to the API. Check your endpoint URL and network.")
        else:
            st.error(f"Error: {e}")

# Clear chat button
if st.session_state["chat_history"]:
    if st.button("Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()
