"""Optional Streamlit web demo for the retrieval-augmented support agent.

Run with:  ./.venv/bin/python -m streamlit run app.py
Requires the deterministic artifacts (agent models + retrieval index built once
via scripts/train_agent_intent.py + scripts/build_retrieval_index.py). The LLM
layer is NOT needed; nothing here calls out to the network except the optional
/chat LLM drafting toggle.

Each interaction shows the full decision trail: predicted intent +
confidence, evidence hits, auto/escalate action with reasons, and the grounded
draft — everything the CLI `agent_chat.py` surfaces, in a browser.
"""
from __future__ import annotations

import html

import streamlit as st

st.set_page_config(page_title="AmazonHelp Support Agent — demo", layout="wide")
st.title("Retrieval-Augmented Support Agent (AmazonHelp)")
st.caption("Deterministic, offline, seed 42 — intent → retrieval → risk escalation → grounded draft.")

from src import config  # noqa: E402
from src.agent import Agent  # noqa: E402
from src.agent.intent import IntentClassifier  # noqa: E402

MODELS = config.DATA_DIR / "retrieval" / "models"


@st.cache_resource
def load_agent():
    model = IntentClassifier.load(MODELS / "intent_weak")
    return Agent.load(intent_dir=MODELS / "intent_weak"), model


agent, _model = load_agent()
examples = [
    "My order hasn't arrived and it's 3 days late, where is it?",
    "Someone used my card to buy things I never ordered!",
    "I received the wrong item, send me the right one please.",
]

col_q, col_r = st.columns([2, 3])
with col_q:
    st.subheader("Customer message")
    query = st.text_area("Message", examples[0], height=120)
    if st.button("Send", type="primary"):
        # Corpus-stamp format (RFC-ish, month-name ordering) so the temporal
        # filter accepts every row: this >= lexicographic max of corpus stamps.
        result = agent.respond(query, created_at="Wed Sep 28 18:06:15 +0000 2016")
        st.session_state["result"] = result

if "result" in st.session_state:
    r = st.session_state["result"]
    with col_r:
        st.subheader("Agent decision")
        st.metric("Intent", r["intent"], delta=None)
        c = st.columns(3)
        c[0].metric("Intent conf", f"{r['intent_prob']:.3f}")
        c[1].metric("Combined conf", f"{r['action_confidence']:.3f}")
        c[2].metric("Evidence hits", len(r["evidence"]))
        st.markdown(f"**Action:** :green[{r['action']}]")
        st.markdown("**Reasons:** " + ", ".join(html.escape(x) for x in r["action_reasons"]))
        st.markdown("---")
        st.markdown("**Draft reply:**")
        st.info(r["draft"])
        st.markdown(f"**Segment:** {r.get('language', '')} · **cues:** {', '.join(r.get('cue_overlap', [])) or 'none'}")

        with st.expander("Top retrieved resolutions (evidence)"):
            for h in r["evidence"][:3]:
                st.markdown(f"- sim **{h['score']:.3f}** — `{h['brand_response'][:180]}`")

st.markdown("---")
st.caption("Demo binds to `data/retrieval/` artifacts; the LLM judge/drafting layer stays optional and is not required here.")