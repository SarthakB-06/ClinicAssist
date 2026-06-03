from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SourceDocument(BaseModel):
    doc_id: str = Field(description="Unique tracking token for the source record chunk.")
    doc_type: str = Field(description="Type of document, e.g., LAB_REPORT, NURSING_NOTE, ER_CHART.")
    timestamp: str = Field(description="Standardized ISO chronological timestamp.")
    raw_content: str = Field(description="The preprocessed text segment within this chunk.")

class SourceMappingTable(BaseModel):
    sentence_index: int
    summary_sentence: str
    verified_source_ids: List[str] = Field(default_factory=list, description="List of document IDs backing this specific claim.")

class DischargeSummaryState(BaseModel):
    # Core Data Inputs
    patient_id: str
    preprocessed_age: int = Field(description="Deterministically computed age to prevent temporal reasoning failures.")
    preprocessed_gender: str = Field(description="Standardized grammatical gender token matching the source chart.")
    chronological_docs: List[SourceDocument] = Field(default_factory=list)
    
    # Extracted Reference Knowledge State
    admission_medications: List[Dict[str, Any]] = Field(default_factory=list)
    discharge_medications: List[Dict[str, Any]] = Field(default_factory=list)
    resolved_diagnoses: List[str] = Field(default_factory=list)
    pending_or_missing_fields: List[str] = Field(default_factory=list, description="Explicitly tracked missing fields.")
    
    # Loop Controls and Iterative Counters
    self_eval_iteration: int = Field(default=0, description="Counter tracking System 2 check iteration cycles.")
    max_eval_cycles: int = Field(default=3, description="Hard cap where self-evaluation returns plateau.")
    is_summary_complete: bool = Field(default=False)
    
    # Guardrails & Audit Safety Logs
    reconciliation_escalation_flags: List[Dict[str, Any]] = Field(default_factory=list, description="Unresolved medication adjustments or diagnostic conflicts.")
    source_attribution_ledger: List[SourceMappingTable] = Field(default_factory=list)
    
    # Final Output Targets
    current_draft: str = Field(default="", description="The evolving clinical document draft text.")
    final_silver_summary: Optional[Dict[str, Any]] = None
    
    # Part 2 Stretch Metrics Tracking
    doctor_edited_gold_summary: str = Field(default="")
    calculated_edit_distance_reward: float = Field(default=0.0)
    step_execution_trace: List[str] = Field(default_factory=list, description="LangSmith backing trace record log.")