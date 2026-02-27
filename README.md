# ✨ Ideafy-agents  
> A multi-agent AI system for structured startup intelligence.

---

> 🎨 **Frontend repository :** **[Ideafy →](https://github.com/hazem-gamal-1/ideafy)**  
> 🌐 **Live demo :** **[ideafy-seven.vercel.app](https://ideafy-seven.vercel.app/)**

---

## 🌌 Overview
**Ideafy** is an AI-powered platform designed to evaluate startup ideas with clarity, structure, and strategic depth.
By orchestrating specialized AI agents, Ideafy transforms a simple idea and supporting knowledge base into a comprehensive, decision-ready analysis.
Upload your documents.  
Describe your startup.  
Receive a complete, structured evaluation — streamed in real time.

---
## 🧠 Architecture
Ideafy is built on a coordinated multi-agent architecture:
User Input (Idea + PDF Knowledge Base)  
↓  
Orchestrator Agent  
↓  
Idea Validation Agent · Legal Analysis Agent · SWOT Analysis Agent  
↓  
Unified Structured Output  
The orchestrator coordinates all sub-agents and merges their outputs into a single structured response.

---
## 🔍 Core Capabilities
### 📊 Idea Validation
- Market potential scoring (0–10)  
- Competition intensity scoring (0–10)  
- Risk identification  
- Executive summary  
### ⚖️ Legal Analysis
- Top legal risks  
- Compliance considerations  
- Recommended action steps  
- Legal summary  
### 🧩 SWOT & Strategic Scenarios
- Strengths  
- Weaknesses  
- Opportunities  
- Threats  
- Forward-looking scenarios  
- Consolidated strategic insight  

---
## 📚 Context-Aware Intelligence
Ideafy converts uploaded PDF documents into semantic embeddings using vector search (ChromaDB + OpenAI embeddings).
Each agent:
1. Determines the context it needs  
2. Retrieves relevant information  
3. Produces structured JSON output  
This ensures analysis is grounded, relevant, and professionally structured.

---
## 🌊 API
**Endpoint**
POST `/analyze`
**Input (Form Data)**
- `file` — PDF knowledge base  
- `prompt` — Startup idea description  
- `actions` — Optional user-provided adjustments  
**Output**
Streaming structured JSON containing:
- Idea validation results  
- Legal analysis  
- SWOT analysis  
- Unified overall summary  

---
## ⚙️ Technology Stack
**Backend**
- FastAPI  
- LangChain  
- LangGraph  
- OpenAI 
- ChromaDB  
- Pydantic  
- Python
- RAG

---
## 🎯 Philosophy
Most idea validation tools provide surface-level feedback.
Ideafy is built differently.
It orchestrates specialized AI agents, grounds them in contextual knowledge, and delivers structured intelligence designed for strategic decision-making.
