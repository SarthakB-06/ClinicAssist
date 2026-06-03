import re
from datetime import datetime 
from typing import List, Dict, Any
import uuid

class ClinicalDocumentParser:
    def __init__(self):
        self.doc_divider_pattern = re.compile(
            r"(CLINICAL PATHOLOGY REPORT|NURSES NOTES|CONSULTATION SHEET|DRUG CHART|ADMISSION RECORD|DISCHARGE CHECK LIST|ER OBSERVATION CHART)", 
            re.IGNORECASE
        )
        self.dob_pattern = re.compile(r"(?:Date of Birth|DOB)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.IGNORECASE)
        self.admit_date_pattern = re.compile(r"Date Of Admission[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.IGNORECASE)
        self.gender_pattern = re.compile(r"Gender[\s:]*(Male|Female|M|F)", re.IGNORECASE)

    def _calculate_age(self, dob_str:str, admit_str: str)-> int:
        """Deterministically calculates age to prevent LLM temporal reasoning failures."""
        try:
            dob = datetime.strptime(dob_str.replace('-', '/'), "%d/%m/%Y")
            admit_date = datetime.strptime(admit_str.replace('-', '/'), "%d/%m/%Y")
            age = admit_date.year - dob.year - ((admit_date.month, admit_date.day) < (dob.month, dob.day))
            return age
        except ValueError:
            return -1 
    
    def parse_raw_text(self, patient_id: str, raw_text: str) -> Dict[str, Any]:
        """Segments raw OCR text and extracts deterministic metadata."""
        
        dob_match = self.dob_pattern.search(raw_text)
        admit_match = self.admit_date_pattern.search(raw_text)
        gender_match = self.gender_pattern.search(raw_text)

        age = -1
        if dob_match and admit_match:
            age = self._calculate_age(dob_match.group(1), admit_match.group(1))
            
        gender = gender_match.group(1).capitalize() if gender_match else "Unknown"

        chunks = self.doc_divider_pattern.split(raw_text)
        documents = []
        
        if chunks[0].strip():
            documents.append({
                "doc_id": str(uuid.uuid4()),
                "doc_type": "GENERAL_ADMISSION",
                "timestamp": admit_match.group(1) if admit_match else "Unknown",
                "raw_content": chunks[0].strip()
            })
            
        for i in range(1, len(chunks), 2):
            doc_type = chunks[i].strip().upper()
            content = chunks[i+1].strip() if i+1 < len(chunks) else ""
            
            if content:
                documents.append({
                    "doc_id": str(uuid.uuid4()),
                    "doc_type": doc_type,
                    "timestamp": admit_match.group(1) if admit_match else "Unknown",
                    "raw_content": content
                })

        return {
            "patient_id": patient_id,
            "preprocessed_age": age,
            "preprocessed_gender": gender,
            "chronological_docs": documents
        }