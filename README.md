Agentic AI: Clinical Discharge Summary Pipeline

An end-to-end, multi-agent AI pipeline built to autonomously generate clinically safe, heavily audited discharge summaries from unstructured, raw patient medical records (PDFs/OCR text).

This system leverages LangGraph to enforce a strict System 1 / System 2 cognitive architecture, prioritizing clinical safety, observability, and automated medication reconciliation over simple text generation.

System Architecture

graph TD
    %% Frontend & Backend
    UI[Streamlit Dashboard] -->|Upload PDF / Paste Text| API[FastAPI Backend]
    
    %% Pre-processing
    subgraph Pre-Processing
        API --> OCR[Vision OCR Ingestor]
        OCR --> Parser[Clinical Document Parser]
        Parser --> Chunking[Chunking & Timestamps]
    end
    
    %% LangGraph State Machine
    subgraph LangGraph Multi-Agent Pipeline
        Chunking --> N1((1. Extraction Node))
        N1 -->|Admit/Discharge Meds & Diagnoses| N2((2. Reconciliation Node))
        
        N2 -->|Silver Draft & Flags| N3((3. System 2 Auditor))
        
        %% The System 2 Loop
        N3 -->|Found Omissions?| N2
        N3 -->|Draft Complete| N4((4. Attribution Node))
        
        N4 -->|Sentence-to-Source Mapping| N5((5. Doctor Eval Node))
    end
    
    %% Output
    N5 -->|Levenshtein Reward & Learned Rules| UI
    
    %% Styling
    classDef frontend fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef graph fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    class UI frontend;
    class N1,N2,N3,N4,N5 graph;

Core Features & Assignment Requirements Met - 
1. 🛡️ Clinical Guardrails (Zero Hallucination Tolerance)

Utilizes strict Pydantic schema enforcement (.with_structured_output()) to extract atomic clinical entities. If a required value (like Patient Age or Admission Date) is missing from the provided context, the LLM is explicitly constrained to output "Not Documented" rather than hallucinating plausible data.
2. 💊 Automated Medication Reconciliation

Silently switching medications between admission and discharge is a leading cause of adverse clinical events. The Reconciliation Node actively compares admission and discharge medication lists. If a medication is added, stopped, or changed without a documented clinical reason in the notes, it generates a high-priority escalation flag.
3. 🔍 System 2 Cognitive Loop (Hallucination Auditing)

Standard LLMs often drop specific quantitative facts (e.g., exact blood glucose levels, specific lab cell counts) when summarizing. This architecture implements a recursive loop where a System 2 Auditor Agent reads the "Silver Draft", compares it against the raw clinical notes, and forces the drafting agent to revise the summary until no critical omissions remain (capped at 3 cycles).
4. 🔗 Observability & Traceability (Source Attribution)

A black-box summary is useless to a clinician. The Attribution Node segments the final draft sentence-by-sentence and maps every single clinical fact back to the specific DOC_ID generated during the parsing phase.
5. 🚀 Stretch Goal: Simulated Doctor Feedback Loop

Implemented a Continuous Learning loop. A Simulated Doctor Node aggressively edits the generated draft into a "Gold" standard. The system calculates the mathematical Levenshtein Edit Distance between the Silver and Gold drafts to output an Agent Accuracy Reward metric. Finally, it extracts generalized rules from the edits to inject into future prompt caches.
🛠️ Tech Stack

    Orchestration: LangGraph, LangChain

    LLM Engine: Gemini 2.5 Flash via Google Generative AI SDK

    Backend: FastAPI, Pydantic (V2)

    Frontend: Streamlit

    Document Processing: pdf2image, Poppler, Pillow


Project Structure - 
ClinicAssist
 ┣ agents/                 # Core LangGraph Architecture
 ┃ ┣ nodes/                # Individual Agent Nodes
 ┃ ┃ ┣ attribution.py      # Maps sentences to source IDs
 ┃ ┃ ┣ doctor_eval.py      # Calculates Levenshtein reward
 ┃ ┃ ┣ extractor.py        # Extracts structured Pydantic data
 ┃ ┃ ┣ reconciliation.py   # Drafts summary & audits meds
 ┃ ┃ ┗ self_eval.py        # System 2 omission auditor
 ┃ ┣ graph.py              # Compiles nodes into LangGraph state machine
 ┃ ┗ state.py              # Master Pydantic state schemas
 ┣ backend/                # API and Data Processing
 ┃ ┣ services/
 ┃ ┃ ┣ document_parser.py  # Chunks raw text and assigns DOC_IDs
 ┃ ┃ ┗ pdf_ingestor.py     # Vision-based OCR for PDFs
 ┃ ┗ main.py               # FastAPI application routing
 ┣ frontend/
 ┃ ┗ app.py                # Streamlit Enterprise Dashboard
 ┣ .env                    # Environment variables (API Keys)
 ┣ requirements.txt        # Python dependencies
 ┗ README.md


Setup & Installation
Prerequisites

    Python 3.10+

    Poppler (Required for PDF conversion)

        Windows: Download Poppler from oschwartz10612/poppler-windows and update the poppler_path in backend/services/pdf_ingestor.py.

        Mac: brew install poppler

        Linux: sudo apt-get install poppler-utils


1. Clone & Install
git clone https://github.com/SarthakB-06/ClinicAssist.git
cd ClinicAssist

# Create a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

2. Environment Variables

Create a .env file in the root directory and add your Google Gemini API key:
GOOGLE_API_KEY="your_api_key_here"
GEMINI_API_KEY="your_api_key_here"

3. Run the Application

You will need two terminal windows to run the backend and frontend simultaneously.

Terminal 1: Start the FastAPI Backend
uvicorn backend.main:app --reload

Terminal 2: Start the Streamlit Frontend
streamlit run frontend/app.py

Navigate to http://localhost:8501 in your browser to interact with the dashboard.