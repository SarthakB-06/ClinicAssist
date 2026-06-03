from pydantic import BaseModel, Field
from typing import List
from agents.state import DischargeSummaryState
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class DoctorEditFeedback(BaseModel):
    gold_draft: str = Field(
        description="The final edited version of the discharge summary."
    )
    extracted_rules: List[str] = Field(
        description="List of 2-3 specific formatting or clinical rules derived from the edits made, to teach the AI for next time."
    )

def calculate_levenshtein_reward(silver: str, gold: str) -> float:
    """Calculates Levenshtein distance and normalizes it to a 0.0 - 1.0 reward score."""
    if len(silver) < len(gold):
        return calculate_levenshtein_reward(gold, silver)
    if len(gold) == 0:
        return 0.0
    
    previous_row = range(len(gold) + 1)
    for i, c1 in enumerate(silver):
        current_row = [i + 1]
        for j, c2 in enumerate(gold):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    distance = previous_row[-1]
    max_len = max(len(silver), len(gold))
    
    reward = 1.0 - (distance / max_len)
    return round(reward, 4)

def doctor_evaluation_node(state: DischargeSummaryState) -> dict:
    """
    LangGraph Node: Simulates a doctor editing the draft, calculates the edit distance reward, 
    and extracts feedback rules for future learning.
    """
    print("--- [NODE: DOCTOR EVALUATION & REWARD] ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    structured_llm = llm.with_structured_output(DoctorEditFeedback)
    
    system_prompt = """
    You are a strict, senior Attending Physician. Your resident AI has just drafted a Discharge Summary.
    You must edit this "Silver Draft" into a "Gold Draft".
    
    YOUR EDITING POLICY:
    1. Make it more concise. Remove fluffy transition words.
    2. Ensure standard medical abbreviations are used consistently (e.g., instead of "intravenous", use "IV").
    3. Keep the Markdown formatting intact, but tighten up the clinical language.
    
    After editing, extract 2-3 specific rules based on the changes you made so the AI can learn for next time.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "SILVER DRAFT TO EDIT:\n\n{draft}")
    ])
    
    eval_chain = prompt | structured_llm
    
    try:
        print("Invoking LLM for simulated doctor edits...")
        result: DoctorEditFeedback = eval_chain.invoke({"draft": state.current_draft})
        
        # Calculate the mathematical reward metric
        reward = calculate_levenshtein_reward(state.current_draft, result.gold_draft)
        
        trace_msg = f"Doctor review complete. Edit Distance Reward: {reward}. Extracted {len(result.extracted_rules)} learning rules."
        
        return {
            "doctor_edited_gold_summary": result.gold_draft,
            "calculated_edit_distance_reward": reward,
            # We save the rules to the trace log to display them in the UI
            "step_execution_trace": state.step_execution_trace + [trace_msg] + [f"LEARNED RULE: {r}" for r in result.extracted_rules]
        }
        
    except Exception as e:
        error_msg = f"Doctor Eval Node API Failure: {str(e)}"
        print(error_msg)
        return {
            "step_execution_trace": state.step_execution_trace + [error_msg]
        }

if __name__ == "__main__":
    import json
    mock_state = DischargeSummaryState(
        patient_id="TEST",
        preprocessed_age=40,
        preprocessed_gender="M",
        current_draft="The patient was given intravenous fluids and oral antibiotics twice a day.",
        step_execution_trace=["Previous steps done."]
    )
    updates = doctor_evaluation_node(mock_state)
    print(f"\nReward: {updates.get('calculated_edit_distance_reward')}")
    print(f"\nGold Draft: {updates.get('doctor_edited_gold_summary')}")