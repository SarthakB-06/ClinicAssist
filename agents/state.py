from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SourceDocument(BaseModel):
    doc_id: str
    doc_type: str
    timestamp: str
    raw_content: str

class SourceMappingTable(BaseModel):
    sentence_index: int
    summary_sentence: str
    verified_source_ids: List[str] = Field(default_factory=list)

class DischargeSummaryState(BaseModel):
    patient_id: str
    preprocessed_age: int
    preprocessed_gender: str
    chronological_docs: List[SourceDocument] = Field(default_factory=list)
    
    admission_medications: List[Dict[str, Any]] = Field(default_factory=list)
    discharge_medications: List[Dict[str, Any]] = Field(default_factory=list)
    resolved_diagnoses: List[str] = Field(default_factory=list)
    pending_or_missing_fields: List[str] = Field(default_factory=list)
    
    self_eval_iteration: int = Field(default=0)
    max_eval_cycles: int = Field(default=3)
    is_summary_complete: bool = Field(default=False)
    
    reconciliation_escalation_flags: List[Dict[str, Any]] = Field(default_factory=list)
    source_attribution_ledger: List[SourceMappingTable] = Field(default_factory=list)
    
    current_draft: str = Field(default="")
    final_silver_summary: Optional[Dict[str, Any]] = None
    
    doctor_edited_gold_summary: str = Field(default="")
    calculated_edit_distance_reward: float = Field(default=0.0)
    step_execution_trace: List[str] = Field(default_factory=list)