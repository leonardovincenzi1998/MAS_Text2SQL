import sqlite3
from typing import List, Optional

class DatabaseManager:
    """Gestisce l'introspezione sicura del database SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def search_tables(self, keyword: Optional[str] = None) -> List[str]:
        """
        Restituisce i nomi delle tabelle, opzionalmente filtrati.
        Esclude le tabelle di sistema di SQLite.
        """
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

    def get_table_ddl(self, table_name: str) -> str:
        """
        Restituisce lo schema DDL. Gestisce errori in modo descrittivo.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            res = cursor.fetchone()
            if res:
                return res[0]
            else:
                #Best Practice: Messaggio di errore che guida l'agente
                return f"ERRORE: La tabella '{table_name}' non esiste. Verifica il nome usando il tool 'list_tables_tool'."
        finally:
            conn.close()