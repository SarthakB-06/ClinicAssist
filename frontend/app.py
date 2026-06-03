import streamlit as st
import requests

st.set_page_config(page_title="AI Discharge Summary Agent", layout="wide")

st.title("🏥 Agentic AI: Discharge Summary Generator")
st.markdown("Generates clinically safe discharge summaries with hallucination audits and source traceability.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Patient Data")
    patient_id = st.text_input("Patient ID", value="PATIENT_002")
    
    upload_mode = st.radio("Input Method", ["Upload PDF (Full Pipeline)", "Paste Raw Text (Fast Test)"])
    
    if upload_mode == "Upload PDF (Full Pipeline)":
        uploaded_file = st.file_uploader("Upload Patient Medical Record (PDF)", type=["pdf"])
        
        if st.button("Run End-to-End Pipeline", type="primary"):
            if not uploaded_file:
                st.error("Please upload a PDF.")
            else:
                # Note: Because OCR takes time, the spinner will spin for a minute or two!
                with st.spinner("Processing PDF (Vision OCR) & Running Agent Pipeline... This may take a few minutes!"):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"patient_id": patient_id}
                        
                        response = requests.post(
                            "http://localhost:8000/api/v1/generate_summary_from_pdf",
                            files=files,
                            data=data
                        )
                        response.raise_for_status()
                        st.session_state.session_agent_data = response.json()
                        st.success("Generation Complete!")
                    except Exception as e:
                        st.error(f"Backend Error: {e}")
                        
    else:
        raw_text = st.text_area("Paste Raw Clinical Notes (OCR Output)", height=300)
        
        if st.button("Generate from Text", type="primary"):
            if not raw_text:
                st.error("Please paste the raw clinical notes first.")
            else:
                with st.spinner("Agent pipeline running... (Extraction ➔ Reconciliation ➔ Audit ➔ Attribution)"):
                    try:
                        response = requests.post(
                            "http://localhost:8000/api/v1/generate_summary_from_text",
                            json={"patient_id": patient_id, "raw_text": raw_text}
                        )
                        response.raise_for_status()
                        st.session_state.session_agent_data = response.json()
                        st.success("Generation Complete!")
                    except Exception as e:
                        st.error(f"Backend Error: {e}")

with col2:
    if "session_agent_data" in st.session_state:
        data = st.session_state.session_agent_data
        
        st.subheader("2. Silver Draft for Review")
        st.markdown(data["draft"])
        
        st.subheader("⚠️ Clinical Safety Flags")
        if data["flags"]:
            for flag in data["flags"]:
                st.warning(f"**{flag['medication_name']} ({flag['issue_type']}):** {flag['description']}")
        else:
            st.success("No unreasoned medication changes detected.")
            
        st.subheader("🔍 Agent Execution Trace")
        with st.expander("View Graph Trace Log"):
            for step in data["trace"]:
                st.write(f"- {step}")