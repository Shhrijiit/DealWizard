# 💻 DEALWIZARD – AI-Powered Laptop Recommendation Assistant

BuyGenie is an intelligent assistant that helps users find the best laptops using natural language queries. It combines large language models (LLMs), semantic search with FAISS, and structured SQL filtering for fast and relevant product recommendations.

---

## 🚀 Features

- 🔍 Understands natural language queries like “lightweight gaming laptops under 1000 euros”
- ⚙️ Uses FAISS for fast vector-based semantic search
- 🧠 Uses Groq-hosted LLaMA 4 for intelligent query understanding
- 📊 Filters laptops by specs like RAM, GPU, weight, and price
- 🛠 SQLite database backend for fast structured queries
- 🤖 Optional: LangChain SQL Agent for querying data conversationally

---

## 🧰 Tech Stack

- Python 3.10+
- [FAISS](https://github.com/facebookresearch/faiss) (vector search)
- [SentenceTransformers](https://www.sbert.net/)
- [LangChain](https://www.langchain.com/)
- [Groq API](https://console.groq.com/)
- SQLite + pandas
- dotenv for config

---

## 📦 Project Structure

project-root/
│
├── db/
│ └── laptops.db # SQLite database with product data
│
├── embeddings/
│ ├── faiss.index # FAISS index for semantic search
│ ├── laptop_dataframe.pkl # DataFrame with laptop details
│ └── id_map.pkl # Mapping between FAISS and DataFrame indices
│
├── llm_query_handler.py # Handles LLM query parsing
├── search_laptops.py # Core logic to run semantic + filter search
├── recommend_with_llm.py # Natural language recommendation generator
├── build_faiss_index.py # Build FAISS index from laptop DB
├── csv_to_sqlite.py # Convert CSV to SQLite
├── sql_agent_assistant.py # Optional: LLM SQL agent via LangChain
│
├── .env # Environment variables (Groq API key)
├── .gitignore
└── README.md

## 🙏 Acknowledgements
This project would not have been possible without the following open-source tools and communities:

Groq – For blazing-fast LLM inference with LLaMA models.

Meta AI – For releasing the LLaMA model family powering natural language understanding.

SentenceTransformers – For providing easy-to-use, high-quality sentence embeddings.

FAISS – For enabling efficient and scalable vector similarity search.

LangChain – For building the SQL agent that powers conversational data access.

OpenAI Python SDK – For convenient LLM interaction handling.

Python community – For the rich ecosystem of data tools (pandas, sqlite3, etc.).
