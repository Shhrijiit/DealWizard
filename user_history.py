import sqlite3
from datetime import datetime
import os

def ensure_column_exists(cursor, table, column, column_type):
    try:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [info[1] for info in cursor.fetchall()]
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    except sqlite3.DatabaseError as e:
        print(f"[ERROR] Failed to ensure column '{column}' exists in '{table}': {e}")

def save_history_to_db(history, db_path="db/user_history.db"):
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)  # Create the directory if not exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Step 1: Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT,
                timestamp TEXT
            )
        """)

        # Step 2: Ensure 'recommendation' column exists
        ensure_column_exists(cursor, "history", "recommendation", "TEXT")

        # Step 3: Insert entries
        for entry in history:
            try:
                cursor.execute("""
                    INSERT INTO history (user_id, query, recommendation, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (
                    entry.get("user_id", "unknown"),
                    entry.get("query", ""),
                    entry.get("recommendation", ""),
                    datetime.now().isoformat()
                ))
            except sqlite3.DatabaseError as e:
                print(f"[ERROR] Failed to insert entry {entry}: {e}")

        conn.commit()

    except sqlite3.Error as e:
        print(f"[ERROR] Database operation failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error occurred: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def get_user_history(user_id, db_path="db/user_history.db"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT query, recommendation, timestamp
            FROM history
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """, (user_id,))

        rows = cursor.fetchall()
        return rows

    except sqlite3.Error as e:
        print(f"[ERROR] Failed to retrieve history for user '{user_id}': {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

# Test saving and retrieving
if __name__ == "__main__":
    sample_history = [
        {"user_id": "user123", "query": "Best gaming laptop", "recommendation": "Try Acer Nitro 5"},
        {"user_id": "user123", "query": "Lightweight laptop for students", "recommendation": "Consider Dell XPS 13"},
        {"user_id": "user123"}  # Intentionally missing fields for testing defaults
    ]
    save_history_to_db(sample_history)
    history = get_user_history("user123")
    for row in history:
        print(row)
