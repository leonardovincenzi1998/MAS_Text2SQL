import re
import os
import json
import sqlglot
from sqlglot import exp, errors
from typing import Tuple, List, Optional, Dict
from xml.parsers.expat import errors
from cv2 import exp
import networkx as nx

# cleans and standardizes table or column names to a canonical format
# removes quotes, brackets, schema prefixes, and converts to lowercase
def get_canonical_name(name: str) -> str:
    if not name:
        return ""
    
    # remove extra spaces around dots
    s = name.strip()
    s = re.sub(r"\s*\.\s*", ".", s)

    # remove common outer quotes or brackets
    for wrap_l, wrap_r in [('"', '"'), ('`', '`'), ('[', ']'), ("'", "'")]:
        if s.startswith(wrap_l) and s.endswith(wrap_r):
            s = s[1:-1].strip()

    # keep only the last part if schema-qualified (e.g., dbo.TableName)
    if "." in s:
        s = s.split(".")[-1].strip()

    # final cleanup of any remaining structural characters
    s = s.strip().strip('"').strip("`").strip("[").strip("]").strip("'")
    
    return s.lower()

# analyzes selected tables and adds necessary bridge tables using shortest path in FK graph
def expand_selection_with_graph(
    selected_tables_real: List[str], 
    schema_json_str: str, 
    root_table_real: Optional[str] = None
) -> List[str]:
    DEBUG_GRAPH = os.getenv("GRAPH_DEBUG", "0") == "1"
    
    def _log_debug(msg: str) -> None:
        if DEBUG_GRAPH:
            print(msg)

    # parse the schema retrieved by the tool
    try:
        schema_list = json.loads(schema_json_str)
    except json.JSONDecodeError:
        return selected_tables_real

    if not schema_list:
        return selected_tables_real

    # build undirected graph because JOINs can go both ways
    G = nx.Graph()
    
    # maps to convert between real and canonical names for safety
    canon_to_real = {}
    
    for table_data in schema_list:
        real_name = table_data.get("table") or table_data.get("table_name")
        canon_name = table_data.get("table_canonical") or get_canonical_name(real_name)
        
        if not canon_name:
            continue
            
        canon_to_real[canon_name] = real_name
        G.add_node(canon_name)
        
    schema_nodes = set(G.nodes)
    
    # add edges based on foreign keys
    for table_data in schema_list:
        real_name = table_data.get("table") or table_data.get("table_name")
        canon_name = table_data.get("table_canonical") or get_canonical_name(real_name)
        
        if not canon_name:
            continue
        
        fks = table_data.get("foreign_keys", [])
        for fk in fks:
            # accept either canonical or real target, and canonicalize
            to_tbl = fk.get("to_table_canonical") or fk.get("to_table_real") or ""
            target_canon = get_canonical_name(str(to_tbl))
            
            if target_canon and target_canon in schema_nodes:
                G.add_edge(canon_name, target_canon)

    _log_debug(f"[GRAPH] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # deterministic normalization to preserve the original order of the LLM list
    ordered_selected_canon = []
    seen = set()
    for name in selected_tables_real:
        c_name = get_canonical_name(name)
        if c_name in G.nodes and c_name not in seen:
            ordered_selected_canon.append(c_name)
            seen.add(c_name)

    if not ordered_selected_canon:
        return selected_tables_real

    # explicitly place root central entity first if present
    root_canon = get_canonical_name(root_table_real) if root_table_real else ""
    if root_canon and root_canon in G.nodes:
        ordered_selected_canon = [root_canon] + [x for x in ordered_selected_canon if x != root_canon]
    else:
        root_canon = ordered_selected_canon[0]

    _log_debug(f"[GRAPH] root_real={root_table_real!r} root_canon={root_canon!r}")
    _log_debug(f"[GRAPH] selected_canon_order={ordered_selected_canon}")

    if len(ordered_selected_canon) < 2:
        return [canon_to_real.get(ordered_selected_canon[0], selected_tables_real[0])]
    
    # filling algorithm via shortest path
    final_set_canon = set(ordered_selected_canon)
    root = ordered_selected_canon[0]

    for target in ordered_selected_canon[1:]:
        try:
            if nx.has_path(G, root, target):
                path = nx.shortest_path(G, root, target)
                final_set_canon.update(path)
                _log_debug(f"[GRAPH] path {root_canon} -> {target}: {path}")
            else:
                _log_debug(f"[GRAPH] no-path {root_canon} -> {target}")
        except nx.NetworkXNoPath:
            _log_debug(f"[GRAPH] NetworkXNoPath {root_canon} -> {target}")

    # deterministic output: LLM tables first, then bridge tables
    final_real_names = []
    out_seen = set()

    for c_name in ordered_selected_canon:
        real = canon_to_real.get(c_name)
        if real and real not in out_seen:
            final_real_names.append(real)
            out_seen.add(real)

    for c_name in sorted(final_set_canon):
        real = canon_to_real.get(c_name)
        if real and real not in out_seen:
            final_real_names.append(real)
            out_seen.add(real)
    
    _log_debug(f"[GRAPH] final_real_names={final_real_names}")
    
    return final_real_names


def validate_ast_and_format(
    raw_sql: str, 
    selected_tables: List[str], 
    selected_columns: Optional[Dict[str, List[str]]] = None
) -> Tuple[str, Optional[str]]:
    """
    Usa sqlglot per analizzare l'AST, validare lo Schema Linking (Tabelle e Colonne)
    e formattare la query.
    """
    try:
        parsed_ast = sqlglot.parse_one(raw_sql, read="sqlite")
        
        # --- 1. MAPPATURA TABELLE E ALIAS ---
        alias_to_table = {}
        used_tables = []
        
        for table in parsed_ast.find_all(exp.Table):
            real_name = table.name
            used_tables.append(real_name)
            
            # Mappa il nome reale verso se stesso (in minuscolo)
            alias_to_table[real_name.lower()] = real_name.lower()
            # Mappa l'alias verso il nome reale (es. 'bm' -> 'benimobili')
            if table.alias:
                alias_to_table[table.alias.lower()] = real_name.lower()

        # --- 2. VALIDAZIONE TABELLE ---
        selected_tables_lower = [t.lower() for t in selected_tables]
        for table in used_tables:
            if table.lower() not in selected_tables_lower:
                return raw_sql, f"AST Error (Schema Linking): La query usa la tabella '{table}', ma non è tra quelle autorizzate {selected_tables}."

        # --- 3. VALIDAZIONE COLONNE ---
        if selected_columns:
            # Creiamo un dizionario tutto minuscolo per confronti sicuri
            allowed_cols_lower = {
                t.lower(): [c.lower() for c in cols] 
                for t, cols in selected_columns.items()
            }
            
            for column in parsed_ast.find_all(exp.Column):
                col_name = column.name.lower()
                
                # Ignoriamo il carattere jolly (*)
                if isinstance(column, exp.Star) or col_name == "*":
                    continue
                    
                # SALVAGUARDIA CHIAVI: Ignoriamo le colonne che iniziano per "id" 
                # perché (come da tua logica in agent.py) vengono sempre passate per i JOIN
                if col_name.startswith("id"):
                    continue

                col_table_alias = column.table.lower() if column.table else None
                
                if col_table_alias:
                    # CASO A: Colonna qualificata (es. bm.Valore)
                    real_table = alias_to_table.get(col_table_alias)
                    
                    if real_table and real_table in allowed_cols_lower:
                        if col_name not in allowed_cols_lower[real_table]:
                            return raw_sql, f"AST Error (Column Linking): La colonna '{column.name}' non è autorizzata per la tabella '{real_table}'."
                else:
                    # CASO B: Colonna non qualificata (es. Valore)
                    # Dobbiamo verificare se esiste in ALMENO UNA delle tabelle autorizzate e usate
                    is_authorized = False
                    for t in used_tables:
                        t_lower = t.lower()
                        if t_lower in allowed_cols_lower and col_name in allowed_cols_lower[t_lower]:
                            is_authorized = True
                            break
                    
                    if not is_authorized:
                         # Se la tabella non ha colonne in allowed_cols_lower (es. tabelle ponte aggiunte dal grafo), 
                         # passiamo oltre, ma se stiamo validando una tabella di cui abbiamo il filtro e non c'è, è errore.
                         return raw_sql, f"AST Error (Column Linking): La colonna '{column.name}' usata nella query non è tra le colonne selezionate dall'Agente 2.5."

        # --- 4. FORMATTAZIONE FINALE ---
        clean_sql = parsed_ast.sql(dialect="sqlite", pretty=True)
        return clean_sql, None

    except errors.ParseError as e:
        return raw_sql, f"AST Syntax Error: La query generata non è sintatticamente valida per SQLite. Dettagli: {str(e)}"
    except Exception as e:
        return raw_sql, f"AST Validation Error: Eccezione durante la validazione dell'AST: {str(e)}"
    
def prune_ddl_ast(raw_ddl: str, allowed_columns: set) -> str:
    """
    Fa il pruning di un DDL (CREATE TABLE) mantenendo solo le colonne autorizzate 
    e i vincoli strutturali (PRIMARY KEY, FOREIGN KEY, ecc.), usando l'AST di SqlGlot.
    """
    try:
        # 1. Parsing del DDL SQLite
        ast = sqlglot.parse_one(raw_ddl, read="sqlite")
        
        # 2. Verifichiamo che sia effettivamente un'istruzione CREATE TABLE
        if isinstance(ast, exp.Create) and isinstance(ast.this, exp.Schema):
            new_expressions = []
            allowed_lower = {c.lower() for c in allowed_columns}
            
            # 3. Iteriamo sugli elementi dentro le parentesi del CREATE TABLE (colonne e vincoli)
            for node in ast.this.expressions:
                if isinstance(node, exp.ColumnDef):
                    col_name = node.name.lower()
                    # Mantieni la colonna se è esplicitamente scelta
                    # o se è una chiave (inizia con 'id') per garantire il corretto JOIN
                    if col_name in allowed_lower or col_name.startswith("id"):
                        new_expressions.append(node)
                else:
                    # Se è un vincolo di tabella (es. CONSTRAINT ... FOREIGN KEY ...)
                    # Lo manteniamo per dare all'LLM il contesto logico delle relazioni
                    new_expressions.append(node)
            
            # 4. Sostituiamo la lista di espressioni nell'AST con quella filtrata
            ast.this.set("expressions", new_expressions)
            
            # 5. Rigeneriamo l'SQL formattato (pretty=True indentifica bene per l'LLM)
            return ast.sql(dialect="sqlite", pretty=True)
        else:
            # Se non è un CREATE (es. un CREATE INDEX allegato), ritorniamo l'originale
            return raw_ddl
            
    except Exception as e:
        print(f"⚠️ Impossibile eseguire il pruning AST sul DDL. Fallback al DDL grezzo. Errore: {e}")
        return raw_ddl