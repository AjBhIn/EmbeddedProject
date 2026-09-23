"""
DocuNode Database Manager (db_manager.py)
-----------------------------------------
Functional helper module to handle local SQLite logging[cite: 1, 3].
Stores document metadata and file paths in docunode.db[cite: 1, 3].
"""

import sqlite3
import os

DB_NAME = "docunode.db"

def init_db():
    """Creates the SQLite database file and scans table if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create scans table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_path TEXT NOT NULL,
            processed_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'COMPLETED'
        );
    """)
    
    conn.commit()
    conn.close()
    print(f"[DB] Database '{DB_NAME}' initialized successfully.")

def log_scan(raw_path, processed_path, status="COMPLETED"):
    """Inserts a new scan record into the database[cite: 3]."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO scans (raw_path, processed_path, status)
            VALUES (?, ?, ?);
        """, (raw_path, processed_path, status))
        
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        print(f"[DB] Successfully logged scan Record #{record_id}")
        return record_id
    except Exception as e:
        print(f"[DB Error] Failed to log scan: {e}")
        return None

def fetch_all_scans():
    """Retrieves all logged document records from the database[cite: 3]."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, raw_path, processed_path, timestamp, status FROM scans ORDER BY id DESC;")
    rows = cursor.fetchall()
    
    conn.close()
    return rows

# Quick test when running this file directly
if __name__ == "__main__":
    init_db()
    log_scan("raw_capture_1.png", "processed_doc_1.png")
    print("Logged Records:", fetch_all_scans())