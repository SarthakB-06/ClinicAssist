# ==========================================
# 1. DEFINE THE OUTPUT SCHEMA (GUARDRAILS)
# ==========================================
from pydantic import BaseModel, Field
from typing import List
from agents.state import DischargeSummaryState, SourceDocument, SourceMappingTable
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.nodes.self_eval import audit_updates
from agents.nodes.extractor import current_state


class AttributionLedger(BaseModel):
    ledger: List[SourceMappingTable] = Field(
        description="A sequential list mapping every sentence in the summary to its source documents."
    )

# ==========================================
# 2. THE ATTRIBUTION NODE FUNCTION
# ==========================================
def attribution_node(state: DischargeSummaryState) -> dict:
    """
    LangGraph Node: Maps every sentence in the final draft back to the source document IDs.
    """
    print("--- [NODE: SOURCE ATTRIBUTION] ---")
    
    # Initialize the LLM (GPT-4o or equivalent)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(AttributionLedger)
    
    # Re-compile context, but this time prominently feature the DOC_ID
    context_blocks = [f"[DOC_ID: {doc.doc_id} | TYPE: {doc.doc_type} | TIME: {doc.timestamp}]\n{doc.raw_content}" for doc in state.chronological_docs]
    full_context = "\n\n".join(context_blocks)
    
    system_prompt = """
    You are an expert Clinical Traceability Agent. 
    Your job is to read a final Discharge Summary and map every single sentence back to the raw source documents.
    
    INSTRUCTIONS:
    1. Break the CURRENT DRAFT down sentence by sentence.
    2. For each sentence, find the specific clinical notes that support the facts in that sentence.
    3. Return the exact sentence, its sequential index, and a list of the 'DOC_ID' strings for the supporting documents.
    4. If a sentence is just formatting (like a Markdown header) or general transition, leave the verified_source_ids list empty.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        CURRENT DRAFT:
        {draft}
        
        RAW CLINICAL NOTES WITH IDs:
        {context}
        """)
    ])
    
    attribution_chain = prompt | structured_llm
    
    try:
        print("Invoking LLM for sentence-level attribution mapping...")
        result: AttributionLedger = attribution_chain.invoke({
            "draft": state.current_draft,
            "context": full_context
        })
        
        trace_msg = f"Attribution complete. Mapped {len(result.ledger)} sentences to source documents."
        
        return {
            "source_attribution_ledger": [mapping.model_dump() for mapping in result.ledger],
            "step_execution_trace": state.step_execution_trace + [trace_msg]
        }
        
    except Exception as e:
        error_msg = f"Attribution Node API Failure: {str(e)}"
        print(error_msg)
        return {
            "step_execution_trace": state.step_execution_trace + [error_msg]
        }
    
if __name__ == "__main__":
    import json
    
    # Mock state to test the attribution node independently
    mock_state = DischargeSummaryState(
        patient_id="TEST_001",
        preprocessed_age=45,
        preprocessed_gender="Male",
        chronological_docs=[
            SourceDocument(
                doc_id="DOC_123",
                doc_type="ER_NOTE",
                timestamp="01/01/2026",
                raw_content="Patient presented with severe dehydration."
            )
        ],
        current_draft="The patient presented with severe dehydration. He was treated with IV fluids.",
        step_execution_trace=["Extraction done", "Drafting done", "Audit done"]
    )

    # Run the Attribution Node
    attribution_updates = attribution_node(mock_state)

    # View the Results
    print("\n--- ATTRIBUTION LEDGER ---")
    ledger = attribution_updates.get("source_attribution_ledger", [])
    print(json.dumps(ledger, indent=2))

    print("\n--- FINAL TRACE LOG ---")
    for step in attribution_updates.get("step_execution_trace", []):
        print(f"- {step}")