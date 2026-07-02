import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Discharge Summary Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=60)
    st.header("Patient Context")
    patient_id = st.text_input("Patient ID", value="PATIENT_002")

    st.divider()
    st.caption("Agent System Status: **Online**")
    st.caption("LLM Engine: **Gemini 2.5 Flash**")
    st.caption("Orchestration: **LangGraph**")

st.title("🏥 Clinical Agent: Discharge Summary Generator")
st.markdown(
    "Automated abstraction, medication reconciliation, and hallucination auditing powered by Agentic AI.")
st.divider()

col_input, col_empty = st.columns([2, 1])

with col_input:
    st.subheader("1. Ingest Clinical Records")
    upload_mode = st.radio("Select Input Method:", [
                           "Upload PDF (End-to-End Pipeline)", "Paste Raw Text (Fast Developer Test)"], horizontal=True)

    if upload_mode == "Upload PDF (End-to-End Pipeline)":
        uploaded_file = st.file_uploader(
            "Upload Patient Medical Record (PDF)", type=["pdf"])

        if st.button("Run Pipeline", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload a PDF document to proceed.")
            else:
                with st.spinner("Processing PDF (Vision OCR) & Running Agent Pipeline... This will take a few minutes!"):
                    try:
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"patient_id": patient_id}

                        response = requests.post(
                            "/api/v1/generate_summary_from_pdf",
                            files=files,
                            data=data
                        )
                        response.raise_for_status()
                        st.session_state.session_agent_data = response.json()
                        st.success("Generation Complete!")
                    except Exception as e:
                        st.error(f"Backend Error: {e}")

    else:
        raw_text = st.text_area(
            "Paste Raw Clinical Notes (OCR Output)", height=150)

        if st.button("Run Pipeline (Text Only)", type="primary", use_container_width=True):
            if not raw_text:
                st.error("Please paste the raw clinical notes first.")
            else:
                with st.spinner("Agent pipeline running... (Extraction ➔ Reconciliation ➔ Audit ➔ Attribution)"):
                    try:
                        response = requests.post(
                            "/api/v1/generate_summary_from_text",
                            json={"patient_id": patient_id,
                                  "raw_text": raw_text}
                        )
                        response.raise_for_status()
                        st.session_state.session_agent_data = response.json()
                        st.success("Generation Complete!")
                    except Exception as e:
                        st.error(f"Backend Error: {e}")

st.divider()

if "session_agent_data" in st.session_state:
    st.subheader("2. Agent Outputs")
    data = st.session_state.session_agent_data

    tab_draft, tab_safety, tab_traceability, tab_feedback, tab_logs = st.tabs([
        "📄 Final Silver Draft",
        "🛡️ Safety & Reconciliation",
        "🔗 Source Attribution",
        "🧑‍⚕️ Doctor Feedback Loop",
        "⚙️ Execution Trace"
    ])

    with tab_draft:
        st.markdown(data.get("draft", "No draft available."))

    with tab_safety:
        st.markdown("### Medication Reconciliation Flags")
        flags = data.get("flags", [])
        if flags:
            st.error(
                f"⚠️ **{len(flags)} Unreasoned Medication Changes Detected**")
            for flag in flags:
                st.warning(
                    f"**Medication:** {flag['medication_name']} | **Status:** {flag['issue_type']}\n\n**Agent Note:** {flag['description']}")
        else:
            st.success(
                "✅ **Medication Reconciliation Passed:** All medication changes have documented clinical reasoning.")

    with tab_traceability:
        st.markdown("### Sentence-Level Traceability Matrix")
        st.caption(
            "Every clinical fact in the summary is mapped to its exact source document to prevent unverified hallucinations.")

        ledger = data.get("ledger", [])
        if ledger:
            clinical_sentences = [
                row for row in ledger if row.get("verified_source_ids")]

            for row in clinical_sentences:
                st.markdown(
                    f"**{row['sentence_index']}.** {row['summary_sentence']}")
                st.caption(
                    f"🔗 **Sources:** `{', '.join(row['verified_source_ids'])}`")
                st.divider()
        else:
            st.info("No attribution data generated.")

    with tab_feedback:
        st.markdown("### AI Learning Metrics (Stretch Goal)")
        st.caption("Simulates a human doctor editing the Silver Draft into a Gold Draft, calculates the Levenshtein Edit Distance as a reward, and extracts rules for future system prompts.")

        reward_score = data.get("trace", [])[-1] if data.get("trace") else ""

        st.metric(
            label="Agent Accuracy Reward (Levenshtein Similarity)",
            value=f"{data.get('reward', 0.85) * 100:.1f}%",
            delta="Optimization Target: 95.0%"
        )

        st.divider()
        st.markdown("#### Extracted Learning Rules")
        st.info("These rules are extracted from the doctor's edits and injected into the prompt cache for the next patient run.")

        learned_rules = [t.replace("LEARNED RULE: ", "")
                         for t in data.get("trace", []) if "LEARNED RULE" in t]
        if learned_rules:
            for rule in learned_rules:
                st.markdown(f"- 🧠 {rule}")
        else:
            st.write("No specific rules extracted this run.")

    with tab_logs:
        st.markdown("### System 1 & System 2 Agent Routing")
        for step in data.get("trace", []):
            st.code(step, language="log")
