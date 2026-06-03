from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.document_parser import ClinicalDocumentParser

app = FastAPI(title="Discharge Summary Agent API", version="1.0")
parser = ClinicalDocumentParser()

class RawDocumentPayload(BaseModel):
    patient_id: str
    raw_text: str

@app.post("/api/v1/preprocess")
async def preprocess_document(payload: RawDocumentPayload):
    """
    Ingests raw OCR text, applies deterministic normalization, 
    and segments into typed chunks.
    """
    try:
        structured_data = parser.parse_raw_text(
            patient_id=payload.patient_id,
            raw_text=payload.raw_text
        )
        return {"status": "success", "data": structured_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
