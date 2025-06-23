# 📊 AI-Powered Project Management Dashboard

This Streamlit application is part of a **capstone / year-end project** exploring **Graph-based Retrieval-Augmented Generation (RAG)** concepts in the context of project management.

It integrates **Neo4j**, **PMBOK knowledge graphs**, and **LLMs** to provide real-time project insights, milestones, tasks, and intelligent recommendations.

---

## 🚀 Features

- 📁 Capture and edit project details, milestones, and tasks
- 📈 Visualize project progress using Gantt and gauge charts
- 📌 Kanban-style task management
- 👥 Track team members and project scope
- 🧠 Integrate structured PMBOK and Reddit-based graph insights
- 🤖 Generate AI summaries and recommendations with LLaMA 3 via Ollama
- 🔐 Uses `.env` variables for secure configuration

---

## 🛠️ Setup Instructions

1. **Clone this repo:**

   ```bash
   git clone https://github.com/your-username/project-management-dashboard.git
   cd project-management-dashboard ```

### Create a .env file:

Create a .env file in the root directory with the following content:
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```


## 📊 Graph Construction
Use the notebook project_pmbok_last_version.ipynb to:

Parse and clean PMBOK and Reddit-based data

Extract graph-based entities and relationships

Push them to Neo4j for semantic reasoning



## 📂 File Structure

```
├── project_manager_app.py             # Main Streamlit application
├── project_pmbok_last_version.ipynb   # Notebook to build the PMBOK + Reddit graph
├── requirements.txt                   # Python dependencies
├── .env                               # Environment configuration (excluded from Git)
├── README.md                          
```

---

## ⚙️ Tech Stack

* **Frontend:** Streamlit + Plotly
* **Backend:** Python
* **Database:** Neo4j (via py2neo)
* **LLM Interface:** LangChain + Ollama (LLaMA 3.2)
* **Graph Insight:** PMBOK + Reddit Knowledge Graphs
* **Optional Vector Store:** FAISS or Qdrant (extendable)

---
