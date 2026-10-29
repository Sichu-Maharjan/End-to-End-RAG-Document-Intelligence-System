import sqlite3
from datetime import datetime

DB_PATH = "../processed/chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_message(username, question, answer):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_history (username, question, answer, timestamp) VALUES (?, ?, ?, ?)",
        (username, question, answer, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_history(username, limit=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT question, answer, timestamp FROM chat_history WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"question": q, "answer": a, "timestamp": t} for q, a, t in rows]

def init_documents_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            filename TEXT PRIMARY KEY,
            num_chunks INTEGER,
            uploaded_by TEXT,
            uploaded_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_document_record(filename, num_chunks, uploaded_by):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO documents (filename, num_chunks, uploaded_by, uploaded_at) VALUES (?, ?, ?, ?)",
        (filename, num_chunks, uploaded_by, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_all_documents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT filename, num_chunks, uploaded_by, uploaded_at FROM documents ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"filename": f, "num_chunks": n, "uploaded_by": u, "uploaded_at": t} for f, n, u, t in rows]

def delete_document_record(filename):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()

init_documents_table()

init_db()