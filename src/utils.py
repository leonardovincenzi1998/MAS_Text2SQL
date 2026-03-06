import re
import os
import json
import sqlglot
from sqlglot import exp, errors
from sqlglot.expressions import Subquery
from typing import Any, Tuple, List, Optional, Dict
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

# transforms the json schema into an optimized pseudo-markdown format for the llm
def format_schema_for_llm(schema_list: list, selected_columns: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Generate the Markdown by reading the pre-calculated strings, applying only the column filter.
    """
    formatted_tables = []
    
    for tbl in schema_list:
        name = tbl.get("table_name") or tbl.get("table", "Unknown")
        desc = tbl.get("description", tbl.get("desc", ""))
        cat_vals = tbl.get("categorical_values", "")
        
        # retrieve the pre-calculated cache and raw columns
        formatted_cols_dict = tbl.get("formatted_columns_dict", {})
        raw_columns = tbl.get("columns", [])
        
        # FUNNEL STEP: Column filter
        cols_to_keep = []
        if selected_columns and name in selected_columns:
            sc_upper = [c.upper() for c in selected_columns[name]]
            cols_to_keep = [c for c in raw_columns if c.upper() in sc_upper or c.upper().startswith("ID")]
        else:
            cols_to_keep = raw_columns

        # directly retrieve the string enriched by the dictionary
        col_tuples = [formatted_cols_dict.get(col, col) for col in cols_to_keep]
            
        # pseudo-markdown construction
        tbl_md = f"### Table: {name}\n"
        
        if desc and desc.strip() and desc != "Unknown": 
            tbl_md += f"Description: {desc}\n"
            
        tbl_md += f"Columns: ( {', '.join(col_tuples)} )\n"
        if cat_vals: 
            tbl_md += f"Notable Values:\n{cat_vals}\n"
            
        formatted_tables.append(tbl_md)
        
    return "\n\n".join(formatted_tables)

def validate_ast_and_format(
    raw_sql: str, 
    selected_tables: List[str], 
    selected_columns: Optional[Dict[str, List[str]]] = None
) -> Tuple[str, Optional[str]]:
    """
    Use sqlglot to analyse the AST, validate the Schema Linking (Tables and Columns)
    and format the query. Includes protections for CTEs and aliases.
    """
    try:
        parsed_ast = sqlglot.parse_one(raw_sql, read="sqlite")
        
        # extract the names of the CTEs (es. WITH TabellaTemp AS ...)
        valid_ctes = {cte.alias.lower() for cte in parsed_ast.find_all(exp.CTE) if cte.alias}
        
        #extract the aliases defined for subqueries (es. FROM (SELECT ... ) AS Subq)
        valid_subqueries = {subq.alias.lower() for subq in parsed_ast.find_all(exp.Subquery) if subq.alias}

        # extract the aliases defined in the SELECT statements (e.g. SUM(Value) AS TotalValue)
        valid_aliases = {alias.alias.lower() for alias in parsed_ast.find_all(exp.Alias) if alias.alias}

        # tables and alias mapping
        alias_to_table = {}
        used_tables = []
        
        for table in parsed_ast.find_all(exp.Table):
            real_name = table.name.lower()
            
            # CTE safeguard, ignore the temporary tables defined in the query
            if real_name in valid_ctes or real_name in valid_subqueries:
                continue
                
            used_tables.append(real_name)
            
            # map the actual name to itself 
            alias_to_table[real_name] = real_name
            # Map the alias to the real name (e.g. “bm” -> “benimobili”)
            if table.alias:
                alias_to_table[table.alias.lower()] = real_name

        # tables validation
        selected_tables_lower = [t.lower() for t in selected_tables]
        for table in used_tables:
            if table not in selected_tables_lower:
                return raw_sql, f"AST Error (Schema Linking): The query uses the table '{table}', but it is not among the authorized tables {selected_tables}."

        # columns validation (only for agent 2.5)
        if selected_columns:
            # create an all-lowercase dictionary for safe comparisons
            allowed_cols_lower = {
                t.lower(): [c.lower() for c in cols] 
                for t, cols in selected_columns.items()
            }
            
            for column in parsed_ast.find_all(exp.Column):
                col_name = column.name.lower()
                
                # ignore the wildcard character (*)
                if isinstance(column, exp.Star) or col_name == "*":
                    continue
                    
                # key safeguarding: Ignore columns beginning with ‘id’
                if col_name.startswith("id"):
                    continue
                    
                # alias safeguard, ignore columns that are actually virtual aliases (e.g. used in ORDER BY)
                if col_name in valid_aliases:
                    continue

                col_table_alias = column.table.lower() if column.table else None
                
                if col_table_alias:
                    # case a: bm.Valore
                    real_table = alias_to_table.get(col_table_alias)
                    
                    if real_table and real_table in allowed_cols_lower:
                        if col_name not in allowed_cols_lower[real_table]:
                            return raw_sql, f"AST Error (Column Linking): The column '{column.name}' is not authorized for the table '{real_table}'."
                else:
                    # case b: Valore
                    is_authorized = False
                    for t in used_tables:
                        if t in allowed_cols_lower and col_name in allowed_cols_lower[t]:
                            is_authorized = True
                            break
                    
                    if not is_authorized:
                         return raw_sql, f"AST Error (Column Linking): The column '{column.name}' used in the query is not among the selected columns for Agent 2.5."

        # final formatting
        clean_sql = parsed_ast.sql(dialect="sqlite", pretty=True)
        return clean_sql, None

    except errors.ParseError as e:
        return raw_sql, f"AST Syntax Error: The generated query is not syntactically valid for SQLite. Details: {str(e)}"
    except Exception as e:
        return raw_sql, f"AST Validation Error: Exception during AST validation: {str(e)}"
    
def prune_ddl_ast(raw_ddl: str, allowed_columns: set) -> str:
    """
    Prune a DDL (CREATE TABLE) keeping only authorised columns
    and structural constraints (PRIMARY KEY, FOREIGN KEY, etc.), using SqlGlot's AST.
    """
    try:
        # 1. parsing the SQLite DDL into an AST
        ast = sqlglot.parse_one(raw_ddl, read="sqlite")
        
        # 2. verify that it is indeed a CREATE TABLE statement
        if isinstance(ast, exp.Create) and isinstance(ast.this, exp.Schema):
            new_expressions = []
            allowed_lower = {c.lower() for c in allowed_columns}
            
            # 3. iterate over the elements inside the CREATE TABLE brackets (columns and constraints)
            for node in ast.this.expressions:
                if isinstance(node, exp.ColumnDef):
                    col_name = node.name.lower()
                    # keep the column if it is explicitly chosen
                    # or if it is a key (starts with “id”) to ensure correct JOIN
                    if col_name in allowed_lower: #or col_name.startswith("id"):
                        new_expressions.append(node)
                else:
                    # if it is a table constraint (e.g. CONSTRAINT ... FOREIGN KEY ...)
                    # keep it to give the LLM the logical context of the relationships.
                    new_expressions.append(node)
            
            # 4. replace the list of expressions in the AST with the filtered one
            ast.this.set("expressions", new_expressions)
            
            # 5. we regenerate the formatted SQL (pretty=True identifies well for the LLM)
            return ast.sql(dialect="sqlite", pretty=True)
        else:
            # if it is not a CREATE (e.g. an attached CREATE INDEX), we return the original.
            return raw_ddl
            
    except Exception as e:
        print(f"⚠️ Impossibile eseguire il pruning AST sul DDL. Fallback al DDL grezzo. Errore: {e}")
        return raw_ddl
    

def get_schema_with_formatted_columns(schema_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pre-calculates Markdown strings for each column (including FK and Sample).
    Adds a “formatted_columns_dict” dictionary to each table in the schema.
    """
    for tbl in schema_list:
        fks = tbl.get("foreign_keys", [])
        samples = tbl.get("column_samples", {})
        
        # 1. pre-calculate fk_map once
        fk_map = {}
        for fk in fks:
            from_col = fk.get("from_column")
            to_tbl = fk.get("to_table_real") or fk.get("to_table_canonical")
            to_col = fk.get("to_column")
            if from_col and to_tbl:
                fk_map[from_col] = f"{to_tbl}.{to_col}" if to_col else to_tbl
                
        # 2. pre-calculates enriched column strings and saves them in cache
        formatted_cols = {}
        for col in tbl.get("columns", []):
            col_str = col
            if col in fk_map:
                col_str += f" [FK->{fk_map[col]}]"
            if col in samples and samples[col]:
                safe_samples = [str(s).replace('\n', ' ').replace('\r', '') for s in samples[col]]
                col_str += f" (Esempi: {', '.join(safe_samples)})"
            
            formatted_cols[col] = col_str
            
        # 3. inject the cache directly into the table metadata
        tbl["formatted_columns_dict"] = formatted_cols
        
    return schema_list

def format_table_metadata_as_sql_comment(tbl_data: dict, allowed_cols: set) -> str:
    """
    Converts the metadata of a table (categorical and sample values) 
    into a block of SQL comments, extracting them from the correct JSON keys.
    """
    if not tbl_data:
        return ""

    table_name = tbl_data.get("table_name", tbl_data.get("table", "Unknown"))
    
    # make the permitted columns case-insensitive for more secure matching
    allowed_lower = {c.lower() for c in allowed_cols}
    
    comments = [f"/* METADATA AND COLUMN PROFILES FOR TABLE '{table_name}':"]
    has_metadata = False
    
    # 1. Retrieve Samples (your “column_samples” field, which is a dict)
    samples = tbl_data.get("column_samples", {})
    if samples:
        for col_name, sample_list in samples.items():
            if col_name.lower() in allowed_lower:
                safe_samples = [str(s).replace('\n', ' ').replace('\r', '') for s in sample_list]
                if safe_samples:
                    comments.append(f" - {col_name} (Samples): [{', '.join(safe_samples)}]")
                    has_metadata = True

    # 2. Recover Categorical Values (your “categorical_values” field, which is a multi-line string)
    cat_vals = tbl_data.get("categorical_values", "")
    if cat_vals and cat_vals.strip():
        valid_cat_lines = [line for line in cat_vals.split('\n') if any(c.lower() in line.lower() for c in allowed_lower)]
        if valid_cat_lines:
            comments.append(" - Categorical Info:")
            for line in valid_cat_lines:
                comments.append(f"   {line}")
            has_metadata = True

    comments.append("*/")
    
    return "\n".join(comments) if has_metadata else ""