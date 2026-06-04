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