from pydantic import BaseModel, Field
from typing import List
from agents.state import DischargeSummaryState, SourceDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI



class System2Evaluation(BaseModel):
    found_omissions: bool = Field(
        description="True if critical clinical facts (e.g., specific lab values, pain scales, treatments) from the source notes are missing in the draft."
    )
    omission_details: List[str] = Field(
        description="List of specific facts that were omitted. Leave empty if none."
    )
    revised_draft: str = Field(
        description="The updated discharge summary incorporating the missing facts. If no omissions were found, output the original draft exactly."
    )

def self_evaluation_node(state: DischargeSummaryState) -> dict:
    """
    LangGraph Node: Audits the current draft against raw sources to catch omissions.
    Limits to a hard cap of 3 cycles.
    """
    iteration = state.self_eval_iteration + 1
    print(f"--- [NODE: SYSTEM 2 EVALUATION] (Cycle {iteration}/{state.max_eval_cycles}) ---")
    
    # Check hard cap
    if iteration > state.max_eval_cycles:
        print("Maximum evaluation cycles reached. Finalizing draft.")
        return {
            "is_summary_complete": True,
            "step_execution_trace": state.step_execution_trace + ["Max System 2 cycles reached. Auto-finalizing."]
        }
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0) 
    structured_llm = llm.with_structured_output(System2Evaluation)
    
    # Re-compile context
    context_blocks = [f"[{doc.doc_type} - {doc.timestamp}]\n{doc.raw_content}" for doc in state.chronological_docs]
    full_context = "\n\n".join(context_blocks)
    
    system_prompt = """
    You are an expert Clinical Auditor (System 2). 
    Your job is to read a drafted Discharge Summary and compare it against the raw chronological clinical notes.
    
    Look for OMISSIONS. Did the draft miss:
    - Specific critical lab values (e.g., exact Sodium, Creatinine, or WBC numbers)?
    - Specific imaging findings?
    - Pain scores or vital signs that indicate clinical trajectory?
    
    If you find omissions:
    1. Set 'found_omissions' to True.
    2. List the specific missing facts in 'omission_details'.
    3. Rewrite the draft in 'revised_draft' to seamlessly weave in these missing facts.
       CRITICAL: When rewriting the draft, you MUST strictly preserve the Markdown formatting. Use double newlines (\n\n) between all bullet points and section headers.
    
    If the draft captures all critical data accurately, set 'found_omissions' to False and return the original draft exactly as it is formatted.
    DO NOT hallucinate data. Only use facts present in the raw clinical notes.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        CURRENT DRAFT:
        {draft}
        
        RAW CLINICAL NOTES:
        {context}
        """)
    ])
    
    eval_chain = prompt | structured_llm
    
    try:
        print("Invoking LLM for clinical auditing...")
        result: System2Evaluation = eval_chain.invoke({
            "draft": state.current_draft,
            "context": full_context
        })
        
        if result.found_omissions:
            trace_msg = f"System 2 Audit Cycle {iteration}: Found omissions: {', '.join(result.omission_details)}. Draft revised."
            is_complete = False
        else:
            trace_msg = f"System 2 Audit Cycle {iteration}: No omissions found. Draft is complete."
            is_complete = True
            
        return {
            "current_draft": result.revised_draft,
            "self_eval_iteration": iteration,
            "is_summary_complete": is_complete,
            "step_execution_trace": state.step_execution_trace + [trace_msg]
        }
        
    except Exception as e:
        error_msg = f"System 2 Node API Failure: {str(e)}"
        print(error_msg)
        return {
            "is_summary_complete": True,
            "step_execution_trace": state.step_execution_trace + [error_msg]
        }
    


if __name__ == "__main__":
    import json
    
    # Mock state to test the auditor independently
    mock_state = DischargeSummaryState(
        patient_id="TEST_001",
        preprocessed_age=45,
        preprocessed_gender="Male",
        chronological_docs=[],
        current_draft="The patient was admitted and treated. He is now better and going home.",
        self_eval_iteration=0,
        max_eval_cycles=3,
        step_execution_trace=["Extraction done", "Drafting done"]
    )

    # Run the System 2 Auditor
    audit_updates = self_evaluation_node(mock_state)

    print(f"\n--- AUDIT COMPLETE? {audit_updates.get('is_summary_complete')} ---")
    print(f"--- TRACE LOG ---")
    print(json.dumps(audit_updates.get("step_execution_trace", [])[-1], indent=2))

    if not audit_updates.get('is_summary_complete'):
        print("\n--- REVISED DRAFT ---")
        print(audit_updates.get("current_draft"))