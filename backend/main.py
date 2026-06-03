from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import sys
import os
import tempfile
import shutil

# Ensure Python can find the agents and backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.graph import build_discharge_summary_graph
from agents.state import DischargeSummaryState, SourceDocument
from backend.services.document_parser import ClinicalDocumentParser
from backend.services.pdf_ingester import LocalMedicalPDFIngestor

app = FastAPI(title="Discharge Summary Agent API")
parser = ClinicalDocumentParser()
graph_app = build_discharge_summary_graph()

class GenerateRequest(BaseModel):
    patient_id: str
    raw_text: str

@app.post("/api/v1/generate_summary_from_text")
async def generate_summary_text(payload: GenerateRequest):
    """Fallback endpoint for fast testing using pre-transcribed text."""
    try:
        parsed_data = parser.parse_raw_text(payload.patient_id, payload.raw_text)
        initial_state = DischargeSummaryState(
            patient_id=parsed_data["patient_id"],
            preprocessed_age=parsed_data["preprocessed_age"],
            preprocessed_gender=parsed_data["preprocessed_gender"],
            chronological_docs=[SourceDocument(**doc) for doc in parsed_data["chronological_docs"]]
        )
        final_state = graph_app.invoke(initial_state)
        
        return {
            "status": "success",
            "draft": final_state.get("current_draft"),
            "flags": final_state.get("reconciliation_escalation_flags", []),
            "trace": final_state.get("step_execution_trace", []),
            "ledger": final_state.get("source_attribution_ledger", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate_summary_from_pdf")
async def generate_summary_pdf(patient_id: str = Form(...), file: UploadFile = File(...)):
    """Full End-to-End Pipeline: OCR -> Parse -> Extract -> Reconcile -> Audit -> Attribute"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    try:
        # 1. Save uploaded file to a temporary file on disk so pdf2image can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # 2. Phase 0: PDF Ingestion (Vision OCR)
        # (Make sure model_name matches what you want to use)
        ingestor = LocalMedicalPDFIngestor(pdf_path=tmp_path, model_name="gemini-2.5-flash") 
        raw_text = ingestor.process_pdf()
        
        # Clean up the temp pdf
        os.remove(tmp_path)
        
        # 3. Phase 1: Preprocessing & Chunking
        parsed_data = parser.parse_raw_text(patient_id, raw_text)
        
        # 4. Initialize Graph State
        initial_state = DischargeSummaryState(
            patient_id=parsed_data["patient_id"],
            preprocessed_age=parsed_data["preprocessed_age"],
            preprocessed_gender=parsed_data["preprocessed_gender"],
            chronological_docs=[SourceDocument(**doc) for doc in parsed_data["chronological_docs"]]
        )
        
        # 5. Execute LangGraph Agents
        final_state = graph_app.invoke(initial_state)
        
        return {
            "status": "success",
            "draft": final_state.get("current_draft"),
            "flags": final_state.get("reconciliation_escalation_flags", []),
            "trace": final_state.get("step_execution_trace", []),
            "ledger": final_state.get("source_attribution_ledger", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))