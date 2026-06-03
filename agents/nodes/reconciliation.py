from pydantic import BaseModel, Field
from typing import List
from agents.state import DischargeSummaryState, SourceDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class ReconciliationFlag(BaseModel):
    medication_name: str = Field(description="The name of the medication involved in the discrepancy.")
    issue_type: str = Field(description="Must be 'Added', 'Stopped', or 'Changed'.")
    description: str = Field(description="Clear explanation of why this change lacks a documented clinical reason.")

class ReconciliationAndDraft(BaseModel):
    reconciliation_flags: List[ReconciliationFlag] = Field(
        description="List of flagged medication discrepancies. Leave empty if all changes have documented reasons."
    )
    silver_draft: str = Field(
        description="The drafted discharge summary formatted in clean Markdown."
    )

def reconciliation_node(state: DischargeSummaryState) -> dict:
    """
    LangGraph Node: Drafts the initial summary and audits medication changes.
    """
    print("--- [NODE: RECONCILIATION & GENERATION] ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0) 
    structured_llm = llm.with_structured_output(ReconciliationAndDraft)
    
    # Re-compile context
    context_blocks = [f"[{doc.doc_type} - {doc.timestamp}]\n{doc.raw_content}" for doc in state.chronological_docs]
    full_context = "\n\n".join(context_blocks)
    
    system_prompt = """
    You are an expert Clinical Reconciliation Agent. Your job is to draft a Discharge Summary and audit medication safety.
    
    TASK 1: MEDICATION RECONCILIATION
    Compare the provided Admission Medications against the Discharge Medications. 
    If a medication was added, stopped, or changed, search the chronological clinical notes for a reason.
    If NO clear clinical reason is documented, you MUST flag it in the `reconciliation_flags`. Do not invent a reason.
    
    TASK 2: DRAFT THE SUMMARY
    Write a structured discharge summary. You MUST use strict Markdown formatting. 
    CRITICAL: You MUST use double newlines (\n\n) between EVERY section header and EVERY bullet point so it renders correctly on a web page. Do not mash text together.
    
    Format exactly like this:
    ## 1. Patient Demographics & Admission/Discharge Dates
    * **Patient Age:** [Value]
    * **Patient Gender:** [Value]
    * **Admission Date:** [Value]
    * **Discharge Date:** [Value]
    
    ## 2. Diagnoses (Principal and Secondary)
    [Bullet points...]
    
    ## 3. Hospital Course
    [Paragraph...]
    
    ## 4. Discharge Medications
    [Bullet points noting changes from admission...]
    
    ## 5. Pending Results
    [Bullet points...]
    
    ## 6. Follow-up Instructions
    [Bullet points...]
    
    ## 7. Discharge Condition
    [Paragraph...]
    
    CRITICAL RULE: If a required field cannot be sourced, explicitly mark it as "Not Documented / Pending Review".
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        Patient Age: {age}
        Patient Gender: {gender}
        
        Admission Medications: {admit_meds}
        Discharge Medications: {discharge_meds}
        Resolved Diagnoses: {diagnoses}
        Pending/Missing Fields: {pending}
        
        Chronological Clinical Notes:
        {context}
        """)
    ])
    
    recon_chain = prompt | structured_llm
    
    try:
        print("Invoking LLM for drafting and reconciliation...")
        result: ReconciliationAndDraft = recon_chain.invoke({
            "age": state.preprocessed_age,
            "gender": state.preprocessed_gender,
            "admit_meds": state.admission_medications,
            "discharge_meds": state.discharge_medications,
            "diagnoses": state.resolved_diagnoses,
            "pending": state.pending_or_missing_fields,
            "context": full_context
        })
        
        flag_count = len(result.reconciliation_flags)
        trace_msg = f"Drafting complete. Identified {flag_count} unreasoned medication changes requiring escalation."
        
        return {
            "current_draft": result.silver_draft,
            "reconciliation_escalation_flags": [flag.model_dump() for flag in result.reconciliation_flags],
            "step_execution_trace": state.step_execution_trace + [trace_msg]
        }
        
    except Exception as e:
        error_msg = f"Reconciliation Node API Failure: {str(e)}"
        print(error_msg)
        return {
            "step_execution_trace": state.step_execution_trace + [error_msg]
        }
    

# 1. Update your existing state with the extraction results you just got
if __name__ == "__main__":
    import json
    
    # Create a mock state to test this node in isolation
    mock_state = DischargeSummaryState(
        patient_id="TEST_001",
        preprocessed_age=45,
        preprocessed_gender="Male",
        chronological_docs=[],
        admission_medications=[{"name": "IV Fluids", "dosage": "1L", "frequency": "Once"}],
        discharge_medications=[{"name": "Oral Antibiotics", "dosage": "500mg", "frequency": "BID"}],
        resolved_diagnoses=["Dehydration"],
        pending_or_missing_fields=[],
        step_execution_trace=["Extraction completed."]
    )
    
    recon_updates = reconciliation_node(mock_state)
    
    print(f"\n--- FLAGS DETECTED: {len(recon_updates.get('reconciliation_escalation_flags', []))} ---")
    print(json.dumps(recon_updates.get("reconciliation_escalation_flags", []), indent=2))
    print("\n--- SILVER DRAFT ---")
    print(recon_updates.get("current_draft", "No draft generated."))