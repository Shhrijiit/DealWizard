import os
import sqlite3
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Configs
DB_PATH = 'db/laptops.db'
TABLE_NAME = 'laptops'

INDEX_SAVE_PATH = 'embeddings/faiss.index'
DF_SAVE_PATH = 'embeddings/laptop_dataframe.pkl'
ID_MAP_SAVE_PATH = 'embeddings/id_map.pkl'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def fetch_laptop_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to fetch laptop data: {e}")
        return pd.DataFrame()

def create_embedding_text(row):
    try:
        return (
            f"{row['Company']} {row['Product']} {row['TypeName']} {row['Inches']} inch, "
            f"{row['Ram']} RAM, {row['OS']}, {row['Weight']}kg, {row['Screen']} "
            f"{row['ScreenW']}x{row['ScreenH']}, Touchscreen: {row['Touchscreen']}, "
            f"IPS: {row['IPSpanel']}, Retina: {row['RetinaDisplay']}, "
            f"{row['CPU_company']} {row['CPU_model']} @ {row['CPU_freq']}GHz, "
            f"{row['PrimaryStorage']} {row['PrimaryStorageType']}, "
            f"{row['SecondaryStorage']} {row['SecondaryStorageType']}, "
            f"{row['GPU_company']} {row['GPU_model']}, Price: {row['Price_euros']} euros"
        )
    except KeyError as e:
        print(f"[WARNING] Missing field while creating embedding text: {e}")
        return ""

def generate_embeddings(texts):
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = model.encode(texts, convert_to_numpy=True).astype('float32')
        return embeddings, model
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        return None, None

def save_index(index):
    try:
        os.makedirs(os.path.dirname(INDEX_SAVE_PATH), exist_ok=True)
        faiss.write_index(index, INDEX_SAVE_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to save FAISS index: {e}")

def save_dataframe(df):
    try:
        os.makedirs(os.path.dirname(DF_SAVE_PATH), exist_ok=True)
        df.to_pickle(DF_SAVE_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to save DataFrame: {e}")

def save_id_map(df):
    try:
        os.makedirs(os.path.dirname(ID_MAP_SAVE_PATH), exist_ok=True)
        id_map = {i: int(df.iloc[i].name) for i in range(len(df))}
        with open(ID_MAP_SAVE_PATH, 'wb') as f:
            pickle.dump(id_map, f)
    except Exception as e:
        print(f"[ERROR] Failed to save ID map: {e}")

def build_faiss_index():
    print("[INFO] Fetching laptop data...")
    df = fetch_laptop_data()
    if df.empty:
        print("[ERROR] No data fetched. Exiting.")
        return

    print("[INFO] Creating text for embeddings...")
    texts = df.apply(create_embedding_text, axis=1).tolist()

    print("[INFO] Generating embeddings...")
    embeddings, model = generate_embeddings(texts)
    if embeddings is None:
        print("[ERROR] Embedding generation failed. Exiting.")
        return

    print(f"[INFO] Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    print("[INFO] Building FAISS index...")
    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
    except Exception as e:
        print(f"[ERROR] Failed to build FAISS index: {e}")
        return

    print("[INFO] Saving index and metadata...")
    save_index(index)
    save_dataframe(df)
    save_id_map(df)

    print(f"[INFO] FAISS index saved at {INDEX_SAVE_PATH}")
    print(f"[INFO] DataFrame saved at {DF_SAVE_PATH}")
    print(f"[INFO] ID map saved at {ID_MAP_SAVE_PATH}")

if __name__ == "__main__":
    build_faiss_index()
