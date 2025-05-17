import pandas as pd
import sqlite3
import os

def csv_to_sqlite(csv_path=r'C:\Users\krish\OneDrive\Documents\Laptop_recommender\laptop_prices.csv.txt', db_path='db/laptops.db'):
    try:
        # Step 1: Check if the CSV file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[ERROR] CSV file not found at: {csv_path}")
        
        # Step 2: Create the db directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Step 3: Read the CSV file
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to read CSV file: {e}")
        
        # Step 4: Check if DataFrame is empty or has issues (you can add more validation depending on your needs)
        if df.empty:
            raise ValueError("[ERROR] The CSV file is empty.")

        # Step 5: Try connecting to SQLite and write the table
        try:
            conn = sqlite3.connect(db_path)
        except sqlite3.Error as e:
            raise ConnectionError(f"[ERROR] Failed to connect to SQLite database: {e}")

        try:
            df.to_sql('laptops', conn, if_exists='replace', index=False)
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to write data to SQLite table: {e}")
        
        # Step 6: Commit the transaction and close the connection
        conn.commit()
        conn.close()
        
        print(f"[INFO] Successfully created SQLite DB at: {db_path}")
    
    except (FileNotFoundError, ValueError, ConnectionError) as error:
        print(error)
    except Exception as e:
        # Catch any other exceptions and print them
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    csv_to_sqlite()
