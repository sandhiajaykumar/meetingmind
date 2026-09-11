# 🧠 MeetingMind — AI Meeting Intelligence & RAG Assistant

MeetingMind is an AI-powered meeting intelligence application that converts meeting recordings into **structured insights** and provides an intelligent **RAG + AI Agent** interface for querying meeting information.

## 🚀 Features

* 🎙️ Upload meeting audio (`MP3`, `WAV`, `M4A`, `MP4`)
* 📝 Automatic meeting transcription using Google Gemini
* 📋 AI-generated meeting summaries
* ✅ Extract important decisions
* 📌 Identify action items, owners, and deadlines
* ⚠️ Detect risks and unresolved issues
* 🧠 Store meeting transcripts as vector embeddings
* 🔎 Semantic search using FAISS
* 📚 Retrieval-Augmented Generation (RAG)
* 🤖 LangGraph-based AI Agent
* 🛠️ Agent tool calling for:

  * Meeting memory search
  * Action-item retrieval
  * Risk analysis
* 💬 Context-aware meeting Q&A
* 🖥️ Interactive Streamlit interface

---

## 🏗️ Architecture

```text
                  🎙️ Meeting Audio
                         │
                         ▼
                 ┌──────────────┐
                 │ Google Gemini│
                 └──────┬───────┘
                        │
                        ▼
              📝 Meeting Analysis
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Summary       Action Items    Risks
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
                 📝 Transcript
                        │
                        ▼
              Text Chunking
                        │
                        ▼
                  Embeddings
                        │
                        ▼
                  🗄️ FAISS
                        │
                        ▼
                 🔎 RAG Retrieval
                        │
                        ▼
                🤖 LangGraph Agent
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
        RAG Tool   Action Tool   Risk Tool
            │           │           │
            └───────────┼───────────┘
                        ▼
                     Gemini
                        │
                        ▼
                  💡 Final Answer
```

---

## 🛠️ Tech Stack

| Technology        | Purpose                          |
| ----------------- | -------------------------------- |
| **Python**        | Application development          |
| **Google Gemini** | Audio analysis and LLM           |
| **LangChain**     | LLM, embeddings and tools        |
| **LangGraph**     | AI agent orchestration           |
| **FAISS**         | Vector database                  |
| **RAG**           | Context-aware question answering |
| **Streamlit**     | Web interface                    |
| **python-dotenv** | Environment variable management  |

---

## 📁 Project Structure

```text
MeetingMind/
│
├── app.py
├── README.md
├── requirements.txt
├── .env
│
├── src/
│   ├── __init__.py
│   ├── rag.py
│   └── agent.py
│
└── data/
    ├── current_meeting.json
    └── vectorstore/
        ├── index.faiss
        └── index.pkl
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/meetingmind.git
cd meetingmind
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install -U streamlit google-genai python-dotenv pydantic langchain langchain-community langchain-core langchain-text-splitters langchain-google-genai langgraph faiss-cpu
```

---

## 🔑 API Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

**Do not commit `.env` to GitHub.**

Add this to `.gitignore`:

```text
.env
venv/
__pycache__/
*.pyc
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Example Usage

### Upload a meeting

Upload a meeting recording such as:

```text
meeting.wav
```

MeetingMind processes the recording and generates:

```text
📝 Transcript

📋 Summary

✅ Decisions

📌 Action Items

⚠️ Risks
```

### Ask questions

Example:

```text
What was the biggest risk?
```

Response:

```text
Retrieval accuracy was identified as the biggest risk.
```

Another example:

```text
What are the pending tasks?
```

Response:

```text
Rahul needs to complete the API integration by Friday.

Priya needs to complete the evaluation dataset by Thursday.
```

---

## 🧠 How RAG Works

MeetingMind uses **Retrieval-Augmented Generation** to answer questions using the uploaded meeting information.

### Step 1 — Chunking

The transcript is divided into smaller chunks.

```text
Transcript
    ↓
Chunk 1
Chunk 2
Chunk 3
...
```

### Step 2 — Embeddings

Each chunk is converted into a numerical vector using Gemini embeddings.

```text
Text → Embedding Vector
```

### Step 3 — Vector Storage

Vectors are stored in **FAISS**.

```text
Meeting Chunks
      ↓
  Embeddings
      ↓
     FAISS
```

### Step 4 — Retrieval

When a user asks a question, semantic similarity search retrieves the most relevant chunks.

### Step 5 — Generation

The retrieved context is passed to Gemini to generate a grounded answer.

```text
Question
   ↓
FAISS Search
   ↓
Relevant Context
   ↓
Gemini
   ↓
Final Answer
```

---

## 🤖 AI Agent

MeetingMind uses **LangGraph** to create an agent capable of selecting different tools based on the user's question.

### Available tools

**Meeting Memory Tool**

Used for questions about meeting discussions and decisions.

**Action Items Tool**

Used for:

```text
tasks
owners
responsibilities
deadlines
```

**Risk Tool**

Used for:

```text
risks
concerns
unresolved issues
```

The agent dynamically selects the appropriate tool and uses Gemini to generate the final response.

---

## 💡 Example Agent Workflow

```text
User:
"What are the pending tasks?"

        ↓

LangGraph Agent

        ↓

Action Item Tool

        ↓

Meeting Data

        ↓

Gemini

        ↓

Final Answer
```

---

## 🔒 Security

Never expose your Gemini API key in source code.

Use:

```env
GEMINI_API_KEY=your_key
```

and keep `.env` in `.gitignore`.

---

## 🚀 Future Improvements

* Multi-meeting conversation memory
* Speaker identification
* Meeting timeline visualization
* Automatic email/task notifications
* Calendar integration
* PostgreSQL/Chroma vector database
* User authentication
* Cloud deployment
* RAG evaluation using precision and recall
* Meeting analytics dashboard
* Export meeting reports as PDF

---

## 👨‍💻 Author

**Sandhi Ajay Kumar**

B.Tech — Computer Science & Engineering
AI & Data Science

---


