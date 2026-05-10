---
title: TalentRoute AI
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# TalentRoute AI — Smart SHL Catalog Matcher

TalentRoute AI is a high-performance conversational assistant designed to help talent acquisition teams, recruiters, and managers match job profiles with the optimal behavioral, skill, and cognitive assessments from the SHL Catalog.

The application leverages a deterministic LangGraph pipeline and a lightweight semantic vector store to provide accurate, context-aware catalog recommendations while maintaining absolute reliability and safety constraints.

---

## ✨ Features

- **Conversational Assistant:** Helps users clarify assessment needs and guides them to the perfect test.
- **Smart Catalog Matching:** Performs hybrid semantic-keyword searches to query the SHL catalog.
- **Constraint Handling:** Multi-turn conversation management with automatic summary boundaries.
- **Production-Ready API:** Standard FastAPI endpoints for easy integration with frontend services.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API Key

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Run Locally
Start the FastAPI server on port **8002** (the default port for the evaluation scripts):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002
```
Open [http://localhost:8002/docs](http://localhost:8002/docs) in your browser to view the interactive API documentation.

---

## 📊 Reproducing Evaluation Results

The system includes a comprehensive evaluation harness to verify recommendation accuracy (Recall@10), latency, and safety constraints.

### 1. Run the Primary Evaluator
This script executes the official evaluation traces (Java Engineer, Financial Analyst, Sales/Marketing refinements, etc.) and computes the final scorecard. 

*Ensure the server is running on port 8002 before starting.*
```bash
python -m scripts.evaluator
```

### 2. Run Robustness & Edge Case Tests
This suite verifies the system's resilience against noise, prompt injection, and multi-turn turn-cap enforcement.
```bash
python scripts/edge_case_tests.py
```

### 🎯 Expected Scorecard (Production Baseline)
When running the evaluator, you should see a scorecard matching these metrics:

```text
==================================================
FINAL SCORECARD
==================================================
Hard Evals:               5/5 passed
Behavior Probes:          28/28 passed
Recall@10:                0.85
Precision@5:              0.49
Latency P95:              < 6.0s
Hallucination Rate:       0.0%
Prompt Injection Res.:    100.0%
Overall Grade:            A
==================================================
```

---

## 🔌 API Endpoints

- **`GET /health`**
  Returns the health status of the application.
  - *Response:* `{"status": "ok"}`

- **`POST /chat`**
  Submit a list of conversation messages to get guided responses and assessment recommendations.
  - *Request Body:*
    ```json
    {
      "messages": [
        {"role": "user", "content": "I need a Java programming test for junior developers."}
      ]
    }
    ```
  - *Response Schema:*
    ```json
    {
      "reply": "Based on your requirement, here is a suitable assessment...",
      "recommendations": [
        {
          "name": "Core Java (Entry Level) (New)",
          "url": "https://www.shl.com/shl-catalog",
          "test_type": "Skills"
        }
      ],
      "end_of_conversation": true
    }
    ```
