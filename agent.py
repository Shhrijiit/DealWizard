import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit, create_sql_agent
from langchain_openai import ChatOpenAI


# Load environment variables safely
try:
    load_dotenv()
    api_key = os.getenv("Groq_Api_Key")
    if not api_key:
        raise ValueError("Groq_Api_Key not found in environment variables.")
except Exception as e:
    raise RuntimeError(f"[ERROR] Could not load API key: {e}")

# Initialize the language model
try:
    llm = ChatOpenAI(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        temperature=0.4
    )
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to initialize language model: {e}")

# Connect to the SQLite database
try:
    db = SQLDatabase.from_uri("sqlite:///db/laptops.db")
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to connect to database: {e}")

# Create toolkit and SQL agent
try:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to create SQL agent: {e}")

# Query function with exception handling
def query_assistant(user_input):
    """
    Executes the assistant agent with the given user input.
    """
    try:
        response = agent_executor.run(user_input)
        return response
    except Exception as e:
        return f"[ERROR] Could not process query: {str(e)}"

# Optional: Testing block
if __name__ == "__main__":
    test_queries = [
        "What are the top 5 cheapest laptops?",
        "List all Apple laptops with more than 8GB RAM",
        "Show me gaming laptops with RTX graphics"
    ]

    for query in test_queries:
        print(f"\n[QUERY] {query}")
        print(query_assistant(query))
