import sqlite3
from typing import List, Optional

# Manages secure introspection of the SQLite database
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    # Opens read-only connection and enforces foreign key constraints
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    # Returns table names excluding SQLite system tables, optionally filtered
    def search_tables(self, keyword: Optional[str] = None) -> List[str]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            params = []
            
            if keyword:
                query += " AND name LIKE ?"
                params.append(f"%{keyword}%")
            
            cursor.execute(query, params)
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    # Returns DDL schema or a descriptive error to guide the LLM agent
    def get_table_ddl(self, table_name: str) -> str:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            res = cursor.fetchone()
            
            if res:
                return res[0]
            else:
                return f"ERROR: Table '{table_name}' does not exist. Verify the name."
        finally:
            conn.close()