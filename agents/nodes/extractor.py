from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.state import DischargeSummaryState, SourceDocument


class Medication(BaseModel):
    name: str = Field(description="Name of the medication.")
    dosage: str = Field(description="Dosage and route, e.g., '1g IV' or '50mg PO'. If not found, output 'Not Documented'.")
    frequency: str = Field(description="Frequency, e.g., 'Daily' or 'BID'. If not found, output 'Not Documented'.")

class ExtractedClinicalData(BaseModel):
    admission_medications: List[Medication] = Field(description="Medications the patient was on upon admission or in the ER.")
    discharge_medications: List[Medication] = Field(description="Medications prescribed at discharge.")
    resolved_diagnoses: List[str] = Field(description="List of all diagnoses confirmed during the hospital stay.")
    pending_or_missing_fields: List[str] = Field(description="List any critical labs (like cultures) marked as 'pending', or missing data.")


# 3. THE EXTRACTOR NODE FUNCTION

def extraction_node(state: DischargeSummaryState) -> dict:
    """
    LangGraph Node: Reads chronological documents and extracts atomic entities.
    Returns a dictionary of state updates.
    """
    print("--- [NODE: EXTRACTION] ---")
    
    # Initialize the LLM (Using GPT-4o-mini or your local equivalent for fast extraction)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # If using Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0) 
    
    # Enforce the Pydantic schema
    structured_llm = llm.with_structured_output(ExtractedClinicalData)
    
    # Prepare the context from Phase 1
    context_blocks = []
    for doc in state.chronological_docs:
        context_blocks.append(f"[{doc.doc_type} - {doc.timestamp}]\n{doc.raw_content}")
    
    full_context = "\n\n".join(context_blocks)
    
    # The strict clinical prompt
    system_prompt = """
    You are an expert Clinical Data Extractor. Your job is to extract specific structured data from the provided clinical notes.
    
    CRITICAL GUARDRAILS:
    1. DO NOT HALLUCINATE. If a piece of information is not explicitly written in the text, you MUST output 'Not Documented'.
    2. If a lab test or culture is sent but the result is not yet available, add it to the `pending_or_missing_fields` list.
    3. Carefully distinguish between medications given IN THE ER/ADMISSION vs. medications prescribed AT DISCHARGE.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here are the patient's chronological documents:\n\n{context}\n\nExtract the required clinical data.")
    ])
    
    # Create the chain and execute
    extractor_chain = prompt | structured_llm
    
    try:
        print("Invoking LLM for structured extraction...")
        result: ExtractedClinicalData = extractor_chain.invoke({"context": full_context})
        
        trace_msg = f"Extraction successful. Found {len(result.resolved_diagnoses)} diagnoses, {len(result.admission_medications)} admit meds, {len(result.discharge_medications)} discharge meds."
        
        # Return the exact fields to update in the LangGraph State
        return {
            "admission_medications": [med.model_dump() for med in result.admission_medications],
            "discharge_medications": [med.model_dump() for med in result.discharge_medications],
            "resolved_diagnoses": result.resolved_diagnoses,
            "pending_or_missing_fields": result.pending_or_missing_fields,
            "step_execution_trace": [trace_msg] # Append to trace log
        }
        
    except Exception as e:
        error_msg = f"Extraction Node API Failure: {str(e)}"
        print(error_msg)
        return {
            "step_execution_trace": [error_msg]
        }


# --- ADD THIS LINE BEFORE YOUR TESTING BLOCK ---
if __name__ == "__main__":
    # 1. Load the text you saved from Phase 0
    with open("patient_2_raw_transcript.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    import sys
    import os

    sys.path.append(os.path.abspath('..'))
    
    # 2. Run it through the Phase 1 Parser we built earlier
    from backend.services.document_parser import ClinicalDocumentParser 
    parser = ClinicalDocumentParser()
    parsed_data = parser.parse_raw_text(patient_id="PATIENT_002", raw_text=raw_text)

    # 3. Initialize the State
    current_state = DischargeSummaryState(
        patient_id=parsed_data["patient_id"],
        preprocessed_age=parsed_data["preprocessed_age"],
        preprocessed_gender=parsed_data["preprocessed_gender"],
        chronological_docs=[SourceDocument(**doc) for doc in parsed_data["chronological_docs"]]
    )

    # 4. Run the Phase 2 Extraction Node!
    state_updates = extraction_node(current_state)

    import json
    print("\n--- NODE OUTPUT ---")
    print(json.dumps(state_updates, indent=2))