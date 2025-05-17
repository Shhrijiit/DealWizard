import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import json

# Load the .env file
load_dotenv()
api_key = os.getenv("Groq_Api_Key")

# Setup OpenAI (Groq) client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Helper to extract and clean the JSON block from LLM response
def clean_response(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        json_str = match.group(0).strip() if match else text
        parsed = json.loads(json_str)

        # Sanitize important_attributes
        if "important_attributes" in parsed:
            for key in ["brand", "model", "use_case"]:
                if parsed["important_attributes"].get(key) is None:
                    parsed["important_attributes"][key] = ""

        return json.dumps(parsed, indent=2)  # Pretty print JSON
    except Exception as e:
        print(f"[ERROR] Failed to clean/parse JSON: {e}")
        return text

def understand_query(user_query):
    prompt = f"""
You are a product search assistant. Extract structured JSON from the user query with keys:
- category (e.g., "laptop")
- intent (e.g., "search", "recommend")
- important_attributes: dictionary with keys (only if present or inferred):
    - price_under
    - price_above
    - lightweight (True/False)
    - brand (e.g., HP, Dell)
    - model
    - use_case (e.g., gaming, office work, student, video editing)

Also infer hidden hardware needs:
- gaming → strong GPU, RAM
- student → lightweight, battery, affordable
- office → lightweight, CPU
- video editing → GPU, CPU, display

Only respond with valid JSON format.

User query: "{user_query}"
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a helpful assistant for understanding product search queries."},
                  {"role": "user", "content": prompt}],
        temperature=0.2
    )

    raw_content = response.choices[0].message.content
    return clean_response(raw_content)

# For testing
if __name__ == "__main__":
    queries = [
        "Find laptops for office work",
        "Find laptops for students",
        "Show me portable HP laptops",
        "Looking for gaming laptops under 60000 rupees",
        "Give me laptops with SSD and Ryzen 7"
    ]

    for q in queries:
        print(f"\n[QUERY] {q}")
        print(understand_query(q))