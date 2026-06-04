# ClinicAssist: Agentic AI Discharge Summary Pipeline

<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" alt="ClinicAssist Logo" width="120">
</p>

<h3 align="center">An end-to-end, multi-agent AI pipeline that autonomously generates clinically safe, heavily audited discharge summaries from unstructured medical records.</h3>

<p align="center">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.128-green?logo=fastapi">
    <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit">
    <img alt="LangChain" src="https://img.shields.io/badge/LangChain-0.2-blueviolet?logo=langchain">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

---

**ClinicAssist** leverages a strict **System 1 / System 2** cognitive architecture using LangGraph to prioritize clinical safety, observability, and automated medication reconciliation over simple text generation. It transforms raw, messy patient records (PDFs/OCR text) into structured, auditable discharge summary drafts ready for clinician review.

## ✨ Core Features

| Feature                               | Description                                                                                                                                                                                                                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🛡️ Clinical Guardrails**            | Utilizes strict Pydantic schemas to enforce zero-hallucination tolerance. If a value is missing from the source, it's marked "Not Documented" instead of being invented.                                                                                                                            |
| **💊 Automated Med Reconciliation**   | Actively compares admission vs. discharge medication lists. Any change without a documented clinical reason generates a high-priority escalation flag, mitigating a common source of adverse events.                                                                                                  |
| **🔍 System 2 Cognitive Loop**        | An internal auditor agent recursively compares the generated draft against source notes to find and correct factual omissions (e.g., specific lab values, vital signs), ensuring high fidelity.                                                                                                      |
| **🔗 Source Attribution**              | Every sentence in the final summary is traced back to the exact source document (`doc_id`) it came from, providing complete transparency and allowing for quick verification by clinicians.                                                                                                         |
| **🚀 Simulated Doctor Feedback**      | A "Simulated Doctor" node edits the AI's draft to a "Gold" standard, calculating a Levenshtein distance reward to quantify accuracy and extracting actionable rules for continuous learning.                                                                                                        |

## 🏛️ System Architecture

The pipeline follows a multi-step, agentic workflow from document ingestion to final, audited output.

```mermaid
graph TD
    subgraph "Input Layer"
        A[/"📄 Raw Medical Record (PDF/Text)"/]
    end

    subgraph "API & Pre-processing (FastAPI)"
        B(1. Vision OCR Ingestor)
        C(2. Document Parser & Chunker)
    end

    subgraph "Agentic Core (LangGraph)"
        D((3. Extraction Agent))
        E((4. Reconciliation & Drafting Agent))
        F((5. System 2 Auditor Agent))
        G((6. Attribution Agent))
        H((7. Doctor Evaluation Agent))
    end

    subgraph "Output & UI (Streamlit)"
        I[/"✅ Audited Discharge Summary"/]
    end

    %% Connections
    A --> B --> C --> D
    D --> E
    E --> F
    F -- "Omissions Found? (Loop)" --> E
    F -- "Draft Complete" --> G
    G --> H
    H --> I

    %% Styling
    classDef api fill:#009688,stroke:#004D40,color:white
    classDef agent fill:#2196F3,stroke:#0D47A1,color:white
    class B,C api
    class D,E,F,G,H agent


## 🛠️ Tech Stack
- **Orchestration**: LangGraph, LangChain
- **LLM**: GEMINI
- **OCR**: PD2Image, Gemini Vision Model
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Data Processing**: Pydantic, Pandas
- **AI Tools**: LangChain, LangGraph


ClinicAssist/
 ┣ 📂 agents/                 # Core LangGraph Architecture
 ┃ ┣ 📂 nodes/                # Individual Agent Functions
 ┃ ┣ 📜 graph.py              # Compiles nodes into the state machine
 ┃ ┗ 📜 state.py              # Pydantic state schemas
 ┣ 📂 backend/                # API and Data Processing
 ┃ ┣ 📂 services/
 ┃ ┃ ┣ 📜 document_parser.py  # Chunks raw text and assigns DOC_IDs
 ┃ ┃ ┗ 📜 pdf_ingestor.py     # Vision-based OCR for PDFs
 ┃ ┗ 📜 main.py               # FastAPI application
 ┣ 📂 frontend/
 ┃ ┗ 📜 app.py                # Streamlit Dashboard UI
 ┣ 📜 .env                    # Environment variables (API Keys)
 ┣ 📜 requirements.txt        # Python dependencies
 ┗ 📜 README.md


## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Poppler**: Required for PDF-to-image conversion.
  - **Windows**: Download [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) and add the `bin` directory to your system's PATH.
  - **macOS**: `brew install poppler`
  - **Linux**: `sudo apt-get install poppler-utils`
- **Google API Key**: You need a Google API key with Gemini API access.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ClinicAssist.git
cd ClinicAssist
```

### 2. Set up the Environment
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 5. Run the Application
The application consists of a FastAPI backend and a Streamlit frontend. Run them in separate terminals.

**Terminal 1: Start the Backend**
```bash
uvicorn backend.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

**Terminal 2: Start the Frontend**
```bash
streamlit run frontend/app.py
```
The Streamlit dashboard will open in your browser at `http://localhost:8501`.

## 📖 Usage
1.  **Launch the Streamlit App**: Open your browser to `http://localhost:8501`.
2.  **Upload a Document**: Use the file uploader to select a patient record (PDF or text file).
3.  **Start Processing**: Click the "Generate Discharge Summary" button.
4.  **Review the Output**: The application will display the generated summary, highlighting key information, medication reconciliation flags, and source attribution for each data point.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.


