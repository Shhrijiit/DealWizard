import os
import re
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from sqlite3 import connect, OperationalError

# ------------------------- Configuration & Logging -------------------------

# Load Groq API key from .env file
load_dotenv()
api_key = os.getenv("Groq_Api_Key")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Validate API key
if not api_key:
    raise EnvironmentError("Groq_Api_Key not found in environment variables")

# Setup OpenAI client for Groq
try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {e}")
    raise

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# ------------------------- Core Recommendation Function -------------------------

def generate_recommendation(user_query, products_df):
    if products_df.empty:
        logging.warning("Empty DataFrame passed to generate_recommendation.")
        return "Sorry, I couldn't find any laptops matching your request."

    # Build formatted laptop list for the prompt
    product_lines = []
    for idx, row in products_df.iterrows():
        try:
            line = (
                f"- {row['Company']} {row['Product']}, {row['Ram']}GB RAM, "
                f"{row['PrimaryStorage']}GB {row['PrimaryStorageType']}, "
                f"{row['GPU_model']} GPU, €{row['Price_euros']}, Weight: {row['Weight']}kg"
            )
            product_lines.append(line)
        except KeyError as ke:
            logging.warning(f"Missing expected column in row {idx}: {ke}")
            continue

    if not product_lines:
        return "Product details were incomplete. No valid options available."

    product_list = "\n".join(product_lines)

    prompt = f"""
You are a tech shopping assistant. A user asked: "{user_query}"

Here are the matching laptops:
{product_list}

Write a helpful natural language recommendation for the user. Focus on the top 2-3 options. Mention highlights like performance, price, portability, or use-case fit (gaming, office, school, etc). Be concise and helpful.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a tech expert who helps users choose laptops."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"LLM request failed: {e}")
        return "Oops, there was a problem generating your recommendation. Please try again later."

# ------------------------- Optional Test Code -------------------------

if __name__ == "__main__":
    try:
        # Attempt to read from the database
        conn = connect("db/laptops.db")
        test_df = pd.read_sql_query("SELECT * FROM laptops LIMIT 3", conn)
        conn.close()
    except FileNotFoundError:
        logging.error("Database file not found.")
        test_df = pd.DataFrame()  # Fallback to empty DataFrame
    except OperationalError as e:
        logging.error(f"SQLite operational error: {e}")
        test_df = pd.DataFrame()

    # Sample query
    user_query = "best laptops for Gamming which weights under 1.5kg"

    try:
        recommendation = generate_recommendation(user_query, test_df)
        print(f"\n[RECOMMENDATION]\n{recommendation}")
    except Exception as e:
        logging.critical(f"Unexpected error during recommendation generation: {e}")
